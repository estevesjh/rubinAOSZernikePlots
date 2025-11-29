# Source code from lsst.rubintv.production.aosUtils
# ------------------------------------------------
# This file is part of rubintv_production.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Any

import batoid
import galsim
import numpy as np
import pandas as pd
from astropy.table import Table
from batoid_rubin import LSSTBuilder

from lsst.afw.cameraGeom import FIELD_ANGLE
from lsst.obs.lsst import LsstCam
from lsst.summit.utils.utils import SIGMATOFWHM
from lsst.ts.ofc import OFCData, StateEstimator
from lsst.ts.ofc.utils.ofc_data_helpers import get_intrinsic_zernikes
from lsst.ts.wep.utils import convertZernikesToPsfWidth, makeDense

__all__ = [
    "makeDataframeFromZernikes",
    "extractWavefrontData",
    "estimateWavefrontDataFromDofs",
    "estimateEllipticities",
    "estimateTelescopeState",
    "getCameraRotatedPositions",
    "parseDofStr",
    "PUPIL_INNER",
    "PUPIL_OUTER",
    "FIELD_RADIUS",
]

PUPIL_INNER = 2.558
PUPIL_OUTER = 4.18
FIELD_RADIUS = 1.75


def makeDataframeFromZernikes(zernikeTable: Table, filterName: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Convert a table of Zernike coefficients into a DataFrame and return the
    rotation matrix.

    Parameters
    ----------
    zernikeTable : `astropy.table.Table`
        Table containing Zernike coefficients with metadata fields
        'nollIndices' and 'rotTelPos'.
    filterName : `str`
        Name of the filter used for the exposure.

    Returns
    -------
    df : `pandas.DataFrame`
        DataFrame with zernikes in OCS, detector name, rotated field angles,
        deviation, and aosFwhm.
    rotMat : `numpy.ndarray`
        2x2 rotation matrix applied to field angles.
    """
    ofcData = OFCData("lsst")
    ofcData.zn_selected = np.array(zernikeTable.meta["nollIndices"])
    rotationAngle = zernikeTable.meta["rotTelPos"]
    rotMat = np.array(
        [
            [np.cos(-rotationAngle), -np.sin(-rotationAngle)],
            [np.sin(-rotationAngle), np.cos(-rotationAngle)],
        ]
    )

    records = []
    for row in zernikeTable:
        zernikesCCS = makeDense(row["zk_CCS"], nollIndices=zernikeTable.meta["nollIndices"])
        zernikesOCS = makeDense(row["zk_OCS"], nollIndices=zernikeTable.meta["nollIndices"])
        intrinsicZernikesSparse = get_intrinsic_zernikes(
            ofcData, filterName.split("_")[0].upper(), [row["detector"]], np.rad2deg(rotationAngle)
        ).squeeze()[ofcData.zn_idx]
        intrinsicZernikes = makeDense(intrinsicZernikesSparse, nollIndices=zernikeTable.meta["nollIndices"])
        zernikesDeviation = zernikesOCS - intrinsicZernikes

        aosFwhm = 1.06 * np.log(1 + np.sqrt(np.sum(np.square(convertZernikesToPsfWidth(zernikesDeviation)))))
        fieldAngles = ofcData.sample_points[row["detector"]]
        rotatedFieldAngles = fieldAngles @ rotMat

        records.append(
            {
                "detector": row["detector"],
                "zernikesCCS": zernikesCCS,
                "zernikesOCS": zernikesOCS,
                "zernikesDeviation": zernikesDeviation,
                "fieldAngles": rotatedFieldAngles,
                "aosFwhm": aosFwhm,
            }
        )

    return pd.DataFrame(records), rotMat


def extractWavefrontData(
    wavefrontResults: pd.DataFrame,
    sourceTable: Table,
    rotMat: np.ndarray,
    zMin: int = 4,
    fieldRadius: float = FIELD_RADIUS,
    kMax: int = 3,
    jMax: int = 28,
    pupilInner: float = PUPIL_INNER,
    pupilOuter: float = PUPIL_OUTER,
) -> dict[str, Any]:
    """Extract Zernike coefficients and FWHM values for measured and
    interpolated data.

    Parameters
    ----------
    wavefrontResults : `pandas.DataFrame`
        DataFrame containing per-detector wavefront results with
        zernikesDeviation, fieldAngles, and aosFwhm.
    sourceTable : `astropy.table.Table`
        Table containing source catalog data.
    rotMat : `numpy.ndarray`
        Rotation matrix to convert angles.
    zMin : `int`, optional
        Minimum Zernike index to consider.
    fieldRadius : `float`, optional
        Field radius, in degrees.
    kMax : `int`, optional
        Maximum Zernike field radial order to use for interpolation.
    jMax : `int`, optional
        Maximum Zernike radial order to use for interpolation.
    pupilInner : `float`, optional
        Inner pupil radius in meters.
    pupilOuter : `float`, optional
        Outer pupil radius in meters.

    Returns
    -------
    result : `dict`
        Dictionary with keys:
        - 'fieldAngles': `numpy.ndarray` measured field angles (deg).
        - 'zksMeasured': `numpy.ndarray` measured Zernikes (padded, shape
        [zMin, jMax+zMin]).
        - 'zksInterpolated': `numpy.ndarray` interpolated Zernikes at rotated
         detector centers.
        - 'rotatedPositions': `numpy.ndarray` rotated field-angle positions of
        detector centers.
        - 'fwhmMeasured': `numpy.ndarray` measured AOS FWHM per detector.
        - 'fwhmInterpolated': `numpy.ndarray` interpolated FWHM at source
        positions.
    """
    # Get rotated positions of the center for each camera detector
    rotatedPositions = getCameraRotatedPositions(rotMat)

    # Retrieve data from Dataframe
    fwhmMeasured = np.vstack(wavefrontResults["aosFwhm"].to_numpy())
    fieldAngles = np.vstack(wavefrontResults["fieldAngles"].to_numpy())
    zernikes = np.vstack(wavefrontResults["zernikesDeviation"].to_numpy())
    zernikesPadded = np.zeros((zernikes.shape[0], zernikes.shape[1] + zMin))
    zernikesPadded[:, zMin : zernikes.shape[1] + zMin] = zernikes

    # Fit a double Zernike to the measured Zernikes with maximum
    # field order of kMax
    basis = galsim.zernike.zernikeBasis(kMax, fieldAngles[:, 0], fieldAngles[:, 1], R_outer=fieldRadius)
    doubleZernikeCoeffs, *_ = np.linalg.lstsq(basis.T, zernikesPadded, rcond=None)
    doubleZernikeCoeffs[0, :] = 0.0  # Need to zero out k=0 term which doesn't have any meaning
    doubleZernikeCoeffs[0, :zMin] = 0.0  # We are not interested in PTT, so we zero them out

    doubleZernikes = galsim.zernike.DoubleZernike(
        doubleZernikeCoeffs,
        uv_inner=0.0,
        uv_outer=fieldRadius,
        xy_inner=pupilInner,
        xy_outer=pupilOuter,
    )

    # Interpolate Zernikes at the rotated positions of the camera detectors
    zksInterpolated = np.zeros((len(rotatedPositions[:, 0]), jMax + 1))
    for idx in range(len(rotatedPositions[:, 0])):
        zksInterpolated[idx, :] = doubleZernikes(rotatedPositions[idx, 0], rotatedPositions[idx, 1]).coef

    # Compute FWHM based on the interpolated Zernikes at the source positions
    fwhmInterpolated = np.zeros(len(sourceTable["aa_x"]))
    for idx in range(len(sourceTable["aa_x"])):
        zks_vec = doubleZernikes(sourceTable["aa_x"][idx], -sourceTable["aa_y"][idx]).coef[zMin:]
        fwhmInterpolated[idx] = np.sqrt(np.sum(convertZernikesToPsfWidth(zks_vec) ** 2))

    return {
        "fieldAngles": fieldAngles,
        "zksMeasured": zernikesPadded,
        "zksInterpolated": zksInterpolated,
        "rotatedPositions": rotatedPositions,
        "fwhmMeasured": fwhmMeasured,
        "fwhmInterpolated": fwhmInterpolated,
    }


def estimateWavefrontDataFromDofs(
    dofState: np.ndarray,
    wavefrontResults: pd.DataFrame,
    sourceTable: Table,
    rotMat: np.ndarray,
    filterName: str,
    batoidFeaDir: str,
    batoidBendDir: str,
    donutBlur: float,
    zMin: int = 4,
    fieldRadius: float = FIELD_RADIUS,
    kMax: int = 6,
    jMax: int = 28,
    obscuration: float = 0.61,
    pupilInner: float = PUPIL_INNER,
    pupilOuter: float = PUPIL_OUTER,
) -> dict:
    """
    Estimate wavefront quantities from a given DOF state using batoid models.

    Parameters
    ----------
    dofState : `numpy.ndarray`
        Array of length 50 representing the AOS DOF state.
    wavefrontResults : `pandas.DataFrame`
        DataFrame used to extract target field angles and detector list.
    sourceTable : `astropy.table.Table`
        Source catalog with fields 'aa_x' and 'aa_y' (degrees) for FWHM
        interpolation.
    rotMat : `numpy.ndarray`
        2x2 rotation matrix to convert field angles.
    filterName : `str`
        Filter name (e.g., 'r', 'i', or including visit info 'r_XXXX' - the
        band prefix is used).
    batoidFeaDir : `str`
        Path to FEA data directory for LSSTBuilder.
    batoidBendDir : `str`
        Path to bend data directory for LSSTBuilder.
    donutBlur : `float`
        Donut blur in arcsec for ellipticity estimation.
    zMin : `int`, optional
        Minimum Zernike index (inclusive) considered when preparing measured
        arrays.
    fieldRadius : `float`, optional
        Field radius (degrees) for the Double-Zernike model.
    kMax : `int`, optional
        Maximum field order for the Double-Zernike.
    jMax : `int`, optional
        Maximum pupil Zernike Noll index for the Double-Zernike.
    obscuration : `float`, optional
        Pupil obscuration ratio eps for batoid Double-Zernike.
    pupilInner : `float`, optional
        Inner pupil radius in meters.
    pupilOuter : `float`, optional
        Outer pupil radius in meters.

    Returns
    -------
    result : `dict`
        Dictionary with keys:
        - 'detector': `list[str]` detector names.
        - 'fieldAngles': `numpy.ndarray` field angles (deg) used for
        evaluation.
        - 'zksEstimated': `numpy.ndarray` estimated Zernikes at measured field
        angles.
        - 'zksMeasured': `numpy.ndarray` measured Zernikes (padded).
        - 'zksInterpolated': `numpy.ndarray` estimated Zernikes at rotated
        detector centers.
        - 'rotatedPositions': `numpy.ndarray` rotated field-angle positions of
        detector centers.
        - 'fwhmMeasured': `numpy.ndarray` measured AOS FWHM per detector.
        - 'fwhmInterpolated': `numpy.ndarray` interpolated FWHM at source
        positions.
    """
    # Get rotated positions of the center for each camera detector
    rotatedPositions = getCameraRotatedPositions(rotMat)

    fwhmMeasured = wavefrontResults["aosFwhm"].to_numpy()
    fieldAngles = np.vstack(wavefrontResults["fieldAngles"].to_numpy())
    zernikes = np.vstack(wavefrontResults["zernikesDeviation"].to_numpy())
    zernikesPadded = np.zeros((zernikes.shape[0], zernikes.shape[1] + zMin))
    zernikesPadded[:, zMin : zernikes.shape[1] + zMin] = zernikes

    ofcData = OFCData("lsst")
    wavelength = ofcData.eff_wavelength[filterName.upper()]

    # Need to fix the signs and unit conversions to use batoid
    dof = -np.array(dofState)  # Make a copy
    dof[[3, 4, 8, 9]] *= 3600  # degrees => arcsec
    dof[[0, 1, 3, 5, 6, 8] + list(range(30, 50))] *= -1  # coordsys

    fiducial = batoid.Optic.fromYaml(f"LSST_{filterName}.yaml")
    telescope = (
        LSSTBuilder(
            fiducial,
            fea_dir=batoidFeaDir,
            bend_dir=batoidBendDir,
        )
        .with_aos_dof(dof)
        .build()
    )

    # Build double Zernike model for the perturbed
    # telescope and the fiducial one
    doubleZernikesPerturbed = (
        batoid.doubleZernike(
            telescope,
            field=np.deg2rad(fieldRadius),
            wavelength=wavelength * 1e-6,
            eps=obscuration,
            jmax=jMax,
            kmax=kMax,
        )
        * wavelength
    )

    doubleZernikesFiducial = (
        batoid.doubleZernike(
            fiducial,
            field=np.deg2rad(fieldRadius),
            wavelength=wavelength * 1e-6,
            eps=obscuration,
            jmax=jMax,
            kmax=kMax,
        )
        * wavelength
    )

    # Generate double zernikes from subtraction of the two
    # (perturbed - fiducial). The fiducial one is small compared
    # to the perturbed one, but not zero.
    doubleZernikeCoeffs = doubleZernikesPerturbed - doubleZernikesFiducial
    doubleZernikes = galsim.zernike.DoubleZernike(
        doubleZernikeCoeffs,
        uv_inner=0,
        uv_outer=fieldRadius,
        xy_inner=pupilInner,
        xy_outer=pupilOuter,
    )

    zksEstimated = np.zeros((fieldAngles.shape[0], jMax + 1))
    for idx in range(fieldAngles.shape[0]):
        zksEstimated[idx, :] = doubleZernikes(fieldAngles[idx, 0], fieldAngles[idx, 1]).coef

    # Interpolate Zernikes at the rotated positions of the camera detectors
    zksInterpolated = np.zeros((len(rotatedPositions[:, 0]), jMax + 1))
    for idx in range(len(rotatedPositions[:, 0])):
        zksInterpolated[idx, :] = doubleZernikes(rotatedPositions[idx, 0], rotatedPositions[idx, 1]).coef

    fwhmInterpolated = np.zeros(len(sourceTable["aa_x"]))
    e1Interpolated = np.zeros(len(sourceTable["aa_x"]))
    e2Interpolated = np.zeros(len(sourceTable["aa_x"]))
    for idx in range(len(sourceTable["aa_x"])):
        fwhmInterpolated[idx], e1Interpolated[idx], e2Interpolated[idx] = estimateEllipticities(
            telescope, sourceTable["aa_x"][idx], -sourceTable["aa_y"][idx], donutBlur, wavelength
        )

    return {
        "detector": wavefrontResults["detector"].to_list(),
        "fieldAngles": fieldAngles,
        "zksEstimated": zksEstimated,
        "zksMeasured": zernikesPadded,
        "zksInterpolated": zksInterpolated,
        "rotatedPositions": rotatedPositions,
        "fwhmMeasured": fwhmMeasured,
        "fwhmInterpolated": fwhmInterpolated,
        "e1Interpolated": e1Interpolated,
        "e2Interpolated": e2Interpolated,
    }


def estimateEllipticities(
    telescope: batoid.Optic,
    thx: float,
    thy: float,
    donutBlur: float,
    wavelength: float,
    pixelSize: float = 10e-6,
    nRad: int = 5,
    nAz: int = 30,
    stampSize: int = 27,
    pupilInner: float = PUPIL_INNER,
    pupilOuter: float = PUPIL_OUTER,
) -> tuple[float, float, float]:
    """Estimate ellipticities from ray tracing through the telescope.

    Parameters
    ----------
    telescope : `batoid.Optic`
        Batoid optic representing the telescope.
    thx : `float`
        Field angle in the x direction (degrees).
    thy : `float`
        Field angle in the y direction (degrees).
    donutBlur : `float`
        Donut blur in arcsec.
    wavelength : `float`
        Wavelength in microns.
    pixelSize : `float`, optional
        Pixel size in meters.
    nRad : `int`, optional
        Number of radial divisions for ray tracing.
    nAz : `int`, optional
        Number of azimuthal divisions for ray tracing.
    stampSize : `int`, optional
        Size of the stamp in pixels.
    pupilInner : `float`, optional
        Inner pupil radius in meters.
    pupilOuter : `float`, optional
        Outer pupil radius in meters.

    Returns
    -------
    fwhm : `float`
        Estimated FWHM in arcsec.
    e1 : `float`
        Estimated ellipticity component e1.
    e2 : `float`
        Estimated ellipticity component e2.
    """
    rv = batoid.RayVector.asPolar(
        optic=telescope,
        wavelength=wavelength * 1e-6,
        theta_x=np.deg2rad(thx),
        theta_y=np.deg2rad(thy),
        nrad=nRad * 3,
        naz=nAz * 3,
        outer=pupilOuter * 0.99,  # Avoid clipping the actual pupil
        inner=pupilInner * 1.01,
    )
    telescope.trace(rv)
    mask = np.logical_not(rv.vignetted)
    xmid = np.mean(rv.x[mask])
    ymid = np.mean(rv.y[mask])

    x, y = rv.x, rv.y
    x -= xmid
    y -= ymid

    # Convolve in a Gaussian
    scale = pixelSize * donutBlur / SIGMATOFWHM / 0.2
    gaussian_var = scale**2

    Ixx = np.nanvar(x) + gaussian_var
    Iyy = np.nanvar(y) + gaussian_var
    Ixy = np.nanmean(x * y) - np.nanmean(x) * np.nanmean(y)

    Ixx *= (1e6 * 0.02) ** 2
    Iyy *= (1e6 * 0.02) ** 2
    Ixy *= (1e6 * 0.02) ** 2

    T = Ixx + Iyy
    e1 = (Ixx - Iyy) / T
    e2 = 2 * Ixy / T

    fwhm = np.sqrt(T / 2 * np.log(256))
    return fwhm, e1, e2


def estimateTelescopeState(
    zernikeTable: Table,
    wavefrontResults: pd.DataFrame,
    filterName: str,
    useDof: str = "0-9,10-16,30-34",
    nKeep: int = 12,
) -> np.ndarray:
    """Estimate the telescope state from wavefront results.

    Parameters
    ----------
    zernikeTable : `astropy.table.Table`
        Table containing Zernike coefficients.
    wavefrontResults : `pandas.DataFrame`
        DataFrame containing per-detector Zernike vectors and detector names.
    filterName : `str`
        Name of the filter used for the exposure.
    useDof : `str`, optional
        Comma-separated integers and/or ranges (e.g., '0-9,10-16,30-34')
        selecting active DOFs.
    nKeep : `int`, optional
        Number of modes to keep in the state estimation truncation.

    Returns
    -------
    dofState : `numpy.ndarray`
        Length-50 array representing the estimated telescope state with
        selected DOFs filled.
    """
    if isinstance(useDof, str):
        newCompDofIdx = parseDofStr(useDof)
    else:
        raise ValueError("useDof must be a string representing integer ranges.")

    ofcData = OFCData("lsst")
    ofcData.zn_selected = np.array(zernikeTable.meta["nollIndices"])
    ofcData.comp_dof_idx = newCompDofIdx
    ofcData.controller["truncation_index"] = nKeep
    stateEstimator = StateEstimator(ofcData)

    zernikesCCS = np.vstack(wavefrontResults["zernikesCCS"].to_numpy())
    detectorNames = wavefrontResults["detector"].to_list()

    out = stateEstimator.dof_state(
        filterName.split("_")[0].upper(),
        zernikesCCS,
        detectorNames,
        np.rad2deg(zernikeTable.meta["rotTelPos"]),
    )

    dofState = np.zeros(50)
    dofState[ofcData.dof_idx] = out
    return dofState


def getCameraRotatedPositions(rotMat: np.ndarray) -> np.ndarray:
    """Get rotated x and y field-angle positions of the camera detectors.

    Parameters
    ----------
    rotMat : `numpy.ndarray`
        Rotation matrix to convert angles.

    Returns
    -------
    rotatedPositions : `numpy.ndarray`
        Array of rotated x and y positions.
    """
    camera = LsstCam().getCamera()
    xPositions: list[float] = []
    yPositions: list[float] = []

    for detector in camera:
        centersDeg = np.rad2deg(detector.getCenter(FIELD_ANGLE))
        xGrid, yGrid = np.meshgrid(centersDeg[0], centersDeg[1])

        xPositions.extend(xGrid.flatten())
        yPositions.extend(yGrid.flatten())

    rotatedPositions = np.array([xPositions, yPositions]).T @ rotMat
    return rotatedPositions


def parseDofStr(dofStr: str) -> dict[str, np.ndarray]:
    """Parse a string representation of integer ranges into a sorted list of
    integers.

    The input string may contain comma-separated integers and/or ranges of the
    form "start-end".

    For example:
        "0-4,10-14" -> [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
        "3,7,9-11"  -> [3, 7, 9, 10, 11]

    Parameters
    ----------
    dofStr : `str`
        A string containing integers and/or integer ranges separated by commas.

    Returns
    -------
    newCompDofIdx : `dict`
        Dictionary with boolean arrays indicating active DOFs per group:
        - 'm2HexPos': `numpy.ndarray` of shape (5,) for M2 hexapod
        - 'camHexPos': `numpy.ndarray` of shape (5,) for Camera hexapod
        - 'M1M3Bend': `numpy.ndarray` of shape (20,) for M1M3 bending modes
        - 'M2Bend': `numpy.ndarray` of shape (20,) for M2 bending modes

    Raises
    ------
    ValueError
        If the string cannot be parsed into integers or ranges of integers.
    """
    dofStr = dofStr.strip()
    useDof: list[int] = []
    for part in dofStr.split(","):
        if "-" in part:
            start, end = [int(p) for p in part.split("-")]
            useDof.extend(range(start, end + 1))
        else:
            useDof.append(int(part))
    useDof = sorted(set(useDof))

    newCompDofIdx = dict(
        m2HexPos=np.full(5, False, dtype=bool),  # M2 hexapod (0–4)
        camHexPos=np.full(5, False, dtype=bool),  # Camera hexapod (5–9)
        M1M3Bend=np.full(20, False, dtype=bool),  # M1M3 bending modes (10–29)
        M2Bend=np.full(20, False, dtype=bool),  # M2 bending modes (30–49)
    )

    # Mark active DOFs
    for idof in useDof:
        if idof < 5:
            # M2 hexapod (x, y, z, tip, tilt)
            newCompDofIdx["m2HexPos"][idof] = True
        elif 5 <= idof < 10:
            # Camera hexapod (x, y, z, tip, tilt)
            newCompDofIdx["camHexPos"][idof - 5] = True
        elif 10 <= idof < 30:
            # M1M3 bending modes (low-order figure control)
            # These modes correct deformations of the
            # primary/tertiary mirror
            newCompDofIdx["M1M3Bend"][idof - 10] = True
        elif 30 <= idof < 50:
            # M2 bending modes (low-order figure control)
            # These modes correct deformations of the secondary mirror
            newCompDofIdx["M2Bend"][idof - 30] = True
    return newCompDofIdx

