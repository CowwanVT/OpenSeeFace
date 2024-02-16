import numpy as np
import cv2
cv2.setNumThreads(6)

def matrix_to_quaternion(m):
    t = 0.0
    q = [0.0, 0.0, 0, 0.0]
    if m[2,2] < 0:
        if m[0,0] > m[1,1]:
            t = 1 + m[0,0] - m[1,1] - m[2,2]
            q = [t, m[0,1]+m[1,0], m[2,0]+m[0,2], m[1,2]-m[2,1]]
        else:
            t = 1 - m[0,0] + m[1,1] - m[2,2]
            q = [m[0,1]+m[1,0], t, m[1,2]+m[2,1], m[2,0]-m[0,2]]
    else:
        if m[0,0] < -m[1,1]:
            t = 1 - m[0,0] - m[1,1] + m[2,2]
            q = [m[2,0]+m[0,2], m[1,2]+m[2,1], t, m[0,1]-m[1,0]]
        else:
            t = 1 + m[0,0] + m[1,1] + m[2,2]
            q = [m[1,2]-m[2,1], m[2,0]-m[0,2], m[0,1]-m[1,0], t]
    return np.array(q) / (pow(t, 0.5)* 0.5)

def landmarks(tensor, crop_info):
    crop_x1, crop_y1, scale_x, scale_y = crop_info

    t_main = tensor[0:66].reshape((66,784))
    t_m = t_main.argmax(1)
    indices = np.expand_dims(t_m, 1)

    t_off_x = np.take_along_axis(tensor[66:132].reshape((66,784)), indices, 1)
    t_off_x = t_off_x.reshape((66,))
    p = np.clip(t_off_x, 0.0000001, 0.9999999)
    t_off_x =  13.9375 * np.log(p / (1 - p))
    t_x = crop_y1 + scale_y * (223. * np.floor(t_m / 28) / 27.+ t_off_x)

    t_off_y = np.take_along_axis(tensor[132:198].reshape((66,784)), indices, 1)
    t_off_y = t_off_y.reshape((66,))
    p = np.clip(t_off_y, 0.0000001, 0.9999999)
    t_off_y =  13.9375 * np.log(p / (1 - p))
    t_y = crop_x1 + scale_x * (223. * np.mod(t_m, 28) / 27. + t_off_y)

    t_conf = np.take_along_axis(t_main, indices, 1).reshape((66,))
    lms = np.stack([t_x, t_y, t_conf], 1)
    lms[np.isnan(lms).any(axis=1)] = [0.,0.,0.]
    return (np.average(t_conf), lms)


#---Things estimate_depth used to do---

def solvePnP(face, im_pts, camera):
    rvec=face.rotation
    tvec=face.translation
    flags=cv2.SOLVEPNP_ITERATIVE
    zeros = np.zeros((4,1))
    contour = face.contour
    return cv2.solvePnP(contour, im_pts, camera, zeros, useExtrinsicGuess=True, rvec=rvec, tvec=tvec, flags=flags)

# Right eyeball
# Eyeballs have an average diameter of 12.5mm
#the distance between eye corners is 30-35mm,
#so a conversion factor of 0.385 can be applied
def rightEye(pts_3d):
    eye_center = (pts_3d[36] + pts_3d[39]) / 2.0
    d_corner = np.linalg.norm(pts_3d[36] - pts_3d[39])
    depth = 0.385 * d_corner
    eye_center[2]-=depth
    return eye_center

def leftEye(pts_3d):
    eye_center = (pts_3d[42] + pts_3d[45]) / 2.0
    d_corner = np.linalg.norm(pts_3d[42] - pts_3d[45])
    depth = 0.385 * d_corner
    eye_center[2]-=depth
    return eye_center

def leftEyePupil(lms, pts_3d, rmat, face,camera, inverse_camera, inverse_rotation):
    d1 = np.linalg.norm(lms[67,0:2] - lms[42,0:2])
    d2 = np.linalg.norm(lms[67,0:2] - lms[45,0:2])
    d = d1 + d2
    pt = (pts_3d[42] * d1 + pts_3d[45] * d2) / d

    reference = rmat.dot(pt)
    reference = reference + face.translation
    reference = camera.dot(reference)
    depth = reference[2]

    pt_3d = [lms[67][0] * depth, lms[67][1] * depth, depth]
    pt_3d = inverse_camera.dot(pt_3d)
    pt_3d = pt_3d - face.translation
    pt_3d = inverse_rotation.dot(pt_3d)
    return pt_3d[:]

def rightEyePupil(lms, pts_3d, rmat, face,camera, inverse_camera, inverse_rotation):
    d1 = np.linalg.norm(lms[66,0:2] - lms[36,0:2])
    d2 = np.linalg.norm(lms[66,0:2] - lms[39,0:2])
    d = d1 + d2
    pt = (pts_3d[36] * d1 + pts_3d[39] * d2) / d

    reference = rmat.dot(pt)
    reference = reference + face.translation
    reference = camera.dot(reference)
    depth = reference[2]

    pt_3d = [lms[66][0] * depth, lms[66][1] * depth, depth]
    pt_3d = inverse_camera.dot(pt_3d)
    pt_3d = pt_3d - face.translation
    pt_3d = inverse_rotation.dot(pt_3d)
    return pt_3d[:]

def points0to66(pts_3d, lms, t_depth_e, inverse_camera, face, inverse_rotation):
    pts_3d[0:66] = np.stack([lms[0:66,0], lms[0:66,1], np.ones((66,))], 1)
    pts_3d[0:66] = pts_3d[0:66] * t_depth_e[0:66]
    pts_3d[0:66] = pts_3d[0:66].dot(inverse_camera.transpose())
    pts_3d[0:66] = pts_3d[0:66] - face.translation
    return pts_3d[0:66].dot(inverse_rotation.transpose())

def calculatePNPerror(lms, t_reference, image_pts ):
    pnp_error = np.power(lms[0:17,0:2] - t_reference[0:17,0:2], 2).sum()
    pnp_error += np.power(lms[30,0:2] - t_reference[30,0:2], 2).sum()

    if np.isnan(pnp_error):
        pnp_error = 9999999.

    pnp_error = pow(pnp_error / (2.0 * image_pts.shape[0]), 0.5)
    return pnp_error

def estimate_depth( face, width, height):
    camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]])
    inverse_camera = np.linalg.inv(camera)
    lms = np.concatenate((face.lms, np.array(face.eye_state)[0:2,1:4]), 0)
    image_pts = lms[face.contourPoints, 0:2]
    pts_3d = np.zeros((70,3), np.float32) #needs to be np.float32

    success, face.rotation, face.translation = solvePnP(face, image_pts, camera)

    if not success:
        face.rotation = [0.0, 0.0, 0.0]
        face.translation = [0.0, 0.0, 0.0]
        return False, np.zeros(4), np.zeros(3), 99999., pts_3d, lms

    rmat, _ = cv2.Rodrigues(face.rotation)

    t_reference = face.face_3d.dot(rmat.transpose())
    t_reference = t_reference + face.translation
    t_reference = t_reference.dot(camera.transpose())

    t_depth = t_reference[:, 2]
    t_depth[t_depth == 0] = 0.000001
    t_depth_e = np.expand_dims(t_depth[:],1)
    t_reference = t_reference[:] / t_depth_e

    inverse_rotation = np.linalg.inv(rmat)

    pts_3d[0:66] = points0to66(pts_3d, lms, t_depth_e, inverse_camera, face, inverse_rotation)
    pts_3d[66,:] = rightEyePupil(lms, pts_3d, rmat, face,camera, inverse_camera, inverse_rotation)
    pts_3d[67,:] = leftEyePupil(lms, pts_3d, rmat, face,camera, inverse_camera, inverse_rotation)
    pts_3d[68] = rightEye(pts_3d)
    pts_3d[69] = leftEye(pts_3d)

    pts_3d[np.isnan(pts_3d).any(axis=1)] = np.array([0.,0.,0.])

    pnp_error = calculatePNPerror(lms, t_reference, image_pts )

    if pnp_error > 300:
        face.fail_count += 1
        if face.fail_count > 5:
            # Something went wrong with adjusting the 3D model
            print("Detected anomaly when 3D fitting face. Resetting")
            face.rotation = np.array([0.0, 0.0, 0.0])
            face.translation = np.array([0.0, 0.0, 0.0])
    else:
        face.fail_count = 0

    euler = cv2.RQDecomp3x3(rmat)[0]

    face.success = True
    face.quaternion = matrix_to_quaternion(rmat)
    face.euler = euler
    face.pnp_error = pnp_error
    face.pts_3d = pts_3d
    face.lms = lms
    face.adjust_3d()
    return face
