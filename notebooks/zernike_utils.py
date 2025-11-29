from __future__ import annotations

import numpy as np
import pandas as pd
import galsim
from astropy.table import Table
from lsst.ts.ofc import OFCData, SensitivityMatrix
from lsst.ts.wep.utils import convertZernikesToPsfWidth
from lsst.summit.utils import ConsDbClient
from aos_utils import getCameraRotatedPositions, PUPIL_INNER, PUPIL_OUTER, FIELD_RADIUS

from lsst.obs.lsst import LsstCam
from lsst.afw.cameraGeom import FIELD_ANGLE
from lsst.ts.ofc.utils.ofc_data_helpers import get_intrinsic_zernikes

CONSDB_URL_DEFAULT = "http://consdb-pq.consdb:8080/consdb"

def getZernikesForDayObs(dayObs: int, consdb_url: str = CONSDB_URL_DEFAULT) -> pd.DataFrame:
    """
    Fetch Zernike data for a specific dayObs from ConsDB.
    
    Parameters
    ----------
    dayObs : `int`
        The dayObs (e.g., 20251124) to fetch.
    consdb_url : `str`, optional
        URL for ConsDB. Defaults to standard internal URL.
        
    Returns
    -------
    df : `pandas.DataFrame`
        DataFrame containing Zernike coefficients and metadata for all visits in the dayObs.
    """
    client = ConsDbClient(consdb_url)
    
    query = f"""
        SELECT
            e.exposure_id AS visit_id,
            e.physical_filter as band,
            e.day_obs AS day_obs,
            e.seq_num AS seq,
            ccdvisit1_quicklook.z4,
            ccdvisit1_quicklook.z5,
            ccdvisit1_quicklook.z6,
            ccdvisit1_quicklook.z7,
            ccdvisit1_quicklook.z8,
            ccdvisit1_quicklook.z9,
            ccdvisit1_quicklook.z10,
            ccdvisit1_quicklook.z11,
            ccdvisit1_quicklook.z12,
            ccdvisit1_quicklook.z13,
            ccdvisit1_quicklook.z14,
            ccdvisit1_quicklook.z15,
            ccdvisit1_quicklook.z16,
            ccdvisit1_quicklook.z17,
            ccdvisit1_quicklook.z18,
            ccdvisit1_quicklook.z19,
            ccdvisit1_quicklook.z20,
            ccdvisit1_quicklook.z21,
            ccdvisit1_quicklook.z22,
            ccdvisit1_quicklook.z23,
            ccdvisit1_quicklook.z24,
            ccdvisit1_quicklook.z25,
            ccdvisit1_quicklook.z26,
            ccdvisit1_quicklook.z27,
            ccdvisit1_quicklook.z28,
            ccdvisit1.detector as detector,
            q.aos_fwhm AS aos_fwhm,
            q.donut_blur_fwhm AS donut_blur_fwhm,
            q.physical_rotator_angle AS rotation_angle,
            e.obs_start
        FROM
            cdb_lsstcam.ccdvisit1_quicklook AS ccdvisit1_quicklook,
            cdb_lsstcam.ccdvisit1 AS ccdvisit1,
            cdb_lsstcam.visit1 AS visit1,
            cdb_lsstcam.visit1_quicklook AS q,
            cdb_lsstcam.exposure AS e
        WHERE
            ccdvisit1.detector IN (191, 192, 195, 196, 199, 200, 203, 204)
            AND ccdvisit1.ccdvisit_id = ccdvisit1_quicklook.ccdvisit_id
            AND ccdvisit1.visit_id = visit1.visit_id
            AND ccdvisit1.visit_id = q.visit_id
            AND ccdvisit1.visit_id = e.exposure_id
            AND e.day_obs = {dayObs}
            AND (e.img_type = 'science' or e.img_type = 'acq' or e.img_type = 'engtest')
            AND e.airmass > 0
            AND e.band != 'none'
        ORDER BY e.seq_num
    """
    
    df = client.query(query).to_pandas()
    
    if df.empty:
        raise RuntimeError(f"No data found in ConsDB for dayObs {dayObs}")
        
    # Convert Zernike columns to a single array column
    zernike_columns = [f"z{i}" for i in range(4, 29)]
    df["zernikes"] = df[zernike_columns].apply(
        lambda row: np.array(row.fillna(0.0).values, dtype=float), axis=1
    )
    
    # Ensure numeric types for FWHM
    df['aos_fwhm'] = pd.to_numeric(df['aos_fwhm'])
    df['donut_blur_fwhm'] = pd.to_numeric(df['donut_blur_fwhm'])
    
    return df


def predictWavefrontFromDofs(
    dofState: np.ndarray,
    wavefrontResults: pd.DataFrame,
    rotMat: np.ndarray,
    sourceTable: Table | None = None,
    zMin: int = 4,
    fieldRadius: float = FIELD_RADIUS,
    kMax: int = 6,
    jMax: int = 28,
    pupilInner: float = PUPIL_INNER,
    pupilOuter: float = PUPIL_OUTER,
    normed: bool = False,
) -> dict:
    """
    Predict wavefront quantities from a given DOF state using the linear
    sensitivity matrix (y = Ax).

    Parameters
    ----------
    dofState : `numpy.ndarray`
        Array of length 50 representing the AOS DOF state.
    wavefrontResults : `pandas.DataFrame`
        DataFrame used to extract target field angles and detector list.
    rotMat : `numpy.ndarray`
        2x2 rotation matrix to convert field angles.
    sourceTable : `astropy.table.Table`, optional
        Source catalog with fields 'aa_x' and 'aa_y' (degrees) for FWHM
        interpolation. If None, FWHM interpolation is skipped.
    zMin : `int`, optional
        Minimum Zernike index (inclusive) considered when preparing measured
        arrays.
    fieldRadius : `float`, optional
        Field radius (degrees) for the Double-Zernike model.
    kMax : `int`, optional
        Maximum field order for the Double-Zernike.
    jMax : `int`, optional
        Maximum pupil Zernike Noll index for the Double-Zernike.
    pupilInner : `float`, optional
        Inner pupil radius in meters.
    pupilOuter : `float`, optional
        Outer pupil radius in meters.
    normed : `bool`, optional
        Whether to apply normalization weights to the sensitivity matrix.
    
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

    # Load OFC Data
    ofcData = OFCData("lsst")
    
    # Initialize Sensitivity Matrix calculator
    dz_sensitivity_matrix = SensitivityMatrix(ofcData)
    
    # Get field angles for the detectors we have
    sensor_name_list = wavefrontResults["detector"].to_list()
    
    # Use the actual (rotated) field angles from wavefrontResults
    field_angles_list = wavefrontResults["fieldAngles"].tolist()
    
    # Evaluate Sensitivity Matrix at these field angles
    # sensitivity_matrix shape: (num_sensors, num_zernikes_total, num_dofs_total)
    sensitivity_matrix = dz_sensitivity_matrix.evaluate(
        field_angles_list, 0.0
    )
    
    # Slice for selected Zernikes and DOFs
    sensitivity_matrix = sensitivity_matrix[:, ofcData.zn_idx, :]
    sensitivity_matrix = sensitivity_matrix[..., ofcData.dof_idx]
    
    if normed:
        # Apply Normalization Weights
        normalization_matrix = np.diag(
            ofcData.normalization_weights[ofcData.dof_idx]
        )
        sensitivity_matrix = sensitivity_matrix @ normalization_matrix
    
    # Predict Zernikes: y = A * x
    # We iterate over sensors to compute y_pred
    zksEstimated = np.zeros((len(fieldAngles), jMax + 1))
    
    for i in range(len(sensor_name_list)):
        A_sensor = sensitivity_matrix[i]
        y_pred_sensor = A_sensor @ dofState[ofcData.dof_idx]
        
        # Map to full Zernike vector
        z_vec = np.zeros(jMax + 1)
        for val, noll in zip(y_pred_sensor, ofcData.zn_selected):
            if noll <= jMax:
                z_vec[noll] = val
        zksEstimated[i, :] = z_vec

    # Fit Double Zernike to Predicted Zernikes for interpolation
    zernikesPredPadded = zksEstimated
    
    basis = galsim.zernike.zernikeBasis(kMax, fieldAngles[:, 0], fieldAngles[:, 1], R_outer=fieldRadius)
    doubleZernikeCoeffs, *_ = np.linalg.lstsq(basis.T, zernikesPredPadded, rcond=None)
    doubleZernikeCoeffs[0, :] = 0.0
    doubleZernikeCoeffs[0, :zMin] = 0.0

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
    if sourceTable is not None:
        fwhmInterpolated = np.zeros(len(sourceTable["aa_x"]))
        e1Interpolated = np.zeros(len(sourceTable["aa_x"]))
        e2Interpolated = np.zeros(len(sourceTable["aa_x"]))
        
        for idx in range(len(sourceTable["aa_x"])):
            zks_vec = doubleZernikes(sourceTable["aa_x"][idx], -sourceTable["aa_y"][idx]).coef[zMin:]
            fwhmInterpolated[idx] = np.sqrt(np.sum(convertZernikesToPsfWidth(zks_vec) ** 2))
            e1Interpolated[idx] = 0.0 
            e2Interpolated[idx] = 0.0
    else:
        fwhmInterpolated = None
        e1Interpolated = None
        e2Interpolated = None

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


def make_wavefront_results(visit_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Construct the wavefrontResults DataFrame from a visit DataFrame.
    
    Parameters
    ----------
    visit_df : `pandas.DataFrame`
        DataFrame containing data for a single visit.
        
    Returns
    -------
    wavefrontResults : `pandas.DataFrame`
        DataFrame containing per-detector Zernike vectors, detector names, 
        rotated field angles, and AOS FWHM.
    """
    # Setup OFC  And Camera
    camera = LsstCam.getCamera()
    ofcData = OFCData("lsst")
    ofcData.zn_selected = np.arange(4, 29)
    filterName = visit_df["band"].iloc[0]
    
    results = {}
    for _, row in visit_df.iterrows():
        # Handle detector name/ID
        det_name = camera[int(row["detector"])].getName()

        # Get rotated field angle for this detector
        fa_rotated, rot_rad = get_field_angle_rotated(row, camera)
        
        # Intrinsic Zernikes
        intrinsic = get_intrinsic_zernikes(
            ofcData, 
            filterName.split("_")[0].upper(), 
            [det_name], 
            np.rad2deg(rot_rad)
        ).squeeze()[ofcData.zn_idx]
        
        zks_meas = row["zernikes"]
        zks_dev = zks_meas - intrinsic
        
        results[det_name] = {
            "fieldAngles": fa_rotated,
            "zernikesCCS": zks_meas,
            "zernikesDeviation": zks_dev,
            "aosFwhm": row["aos_fwhm"],
            "rot_tel_rad": rot_rad
        }
    
    wavefrontResults = pd.DataFrame(results).T
    wavefrontResults.reset_index(inplace=True)
    wavefrontResults.rename(columns={"index": "detector"}, inplace=True)
    return wavefrontResults

def get_field_angle_rotated(row: pd.Series, camera: LsstCam) -> tuple[np.ndarray, float]:
    """
    Compute the rotated field angle for a detector and return rotation angle.
    
    This function retrieves the nominal field angle for a detector's center position
    and applies a per-detector rotation transformation to account for physical rotation
    of the camera or instrument.
    
    Parameters
    ----------
    row : `pandas.Series`
        A pandas Series containing detector data with fields:
        - 'detector' : detector ID (int)
        - 'rotation_angle' : physical rotation angle in degrees (float)
    camera : `lsst.obs.lsst.LsstCam`
        The LSST camera object providing access to detector geometry.
    
    Returns
    -------
    fa_rotated : `numpy.ndarray`
        The rotated field angle coordinates in degrees as a 1D array of shape (2,)
        containing [x, y] field angle positions.
    rot_rad : `float`
        The rotation angle converted to radians.
    """
    # Handle detector name/ID
    det_id = int(row["detector"])
    det = camera[det_id]

    # Field Angle (Nominal)
    bbox = det.getBBox()
    centerX, centerY = bbox.getCenter()
    transform = camera.getTransform(det.getNativeCoordSys(), FIELD_ANGLE)
    p_out = transform.getMapping().applyForward(np.vstack((centerX, centerY)))
    fa = np.rad2deg(p_out.flatten())
    
    # Apply Rotation (Per Detector)
    rot_rad = np.deg2rad(row["rotation_angle"])
    rot_mat = np.array([
        [np.cos(rot_rad), np.sin(rot_rad)],
        [-np.sin(rot_rad), np.cos(rot_rad)]
    ])
    fa_rotated = fa @ rot_mat
    return fa_rotated, rot_rad
