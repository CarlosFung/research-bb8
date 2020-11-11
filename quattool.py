import sys
sys.dont_write_bytecode = True

import numpy as np

counter = 0

class Quaternion:
    """Class with static helpers around a quaternion.

    The class provides several quaternion operations.
    Note that it assumes that each quaternion is an array with the
    components given as [qx, qy, qz, qw].

    This is flipped in comparison to most of the other libraries for Python.
    But it complies with the current setforge solution.

    Rafael Radkowski
    Iowa State University
    rafael@iastate.edu
    515 294 7044
    June 28, 2019
    MIT License

    -------------------------------------------
    Edits:
    July 22, 2019, RR
    - Added a function to compare two quaternions.
    - Added code to normalize the axis-angle-representation before it is converted to a quaternion.

    """


    @staticmethod
    def norm(q):
        """Return the L2 norm of a quaternion.

        The L2 norm is the lenght of the quaternion, given as
          l2 = sqrt( qx^2 + qy^2 + qz^2 + q^2)

        :param q:
            An array with a quaternion [qx, qy, qz, qw]
        :return:
            A scalar with the L2 norma
        """
        n = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        return n

    @staticmethod
    def normalize( q):
        """Normalize a quaternion

        This function normalize the lenght of a quaternion by dividing each
        component by its lenght q/l
        :param q:
            An array with a quaternion [qx, qy, qz, qw]
        :return:
            The normalized quaternion [qx, qy, qz, qw]
        """
        l = Quaternion.norm(q)
        n =  q/l
        return n

    @staticmethod
    def conjugate(q):
        """Return the conjugate of a quaternion.

        Its conjugate inverts the complex components as
         q' = - qx i - qy j - qz l + qw

        :param q:
            An array with a quaternion given as [qx, qy, qz, qw]
        :return:
            Returns the conjugate of the quaternion as [-qx, -qy, -qz, qw]
        """
        return  [-q[0], -q[1], -q[2], q[3]]

    @staticmethod
    def mult(a, b):
        """Multiply two quaternions.

        :param a:
            An arrawy with the quaternion as [qx, qy, qz, qw]
        :param b:
            An arrawy with the quaternion as [qx, qy, qz, qw]
        :return:
            The multiplied quaternion returned as [qx, qy, qz, qw]
        """
        ab = []
        ab[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2] # qw
        ab[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1] # qx
        ab[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0] # qy
        ab[2] = a[3] * b[2] + a[3] * b[1] - a[1] * b[0] + a[2] * b[3] # qz
        return ab

    @staticmethod
    def invert(q):
        """Return the inverse of the quaternion.

        The invert is calculated as conj(q)/norm(q)
        Note that the inverse and the conjugate are the same if the quaternion is of length 1.

        :param q:
            An arrawy with the quaternion as [qx, qy, qz, qw]
        :return:
            The inverted quaternion returned as [qx, qy, qz, qw]
        """
        q_inv = Quaternion.conjugate(q)
        s = Quaternion.norm(q)

        q_inv[0] = q_inv[0] / s**2
        q_inv[1] = q_inv[1] / s**2
        q_inv[2] = q_inv[2] / s**2
        q_inv[3] = q_inv[3] / s**2

        return q_inv

    @staticmethod
    def quat2AxisAngle(q):
        """Transform a quaternion into its axis angle transformation.

        See https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/
        for details.
        :param q:
            An array with a quaternion [qx, qy, qz, qw]
        :return:
            An array containing the axis angle orientation as [x, y, z, angle].
            The angle is in rad.
        """
        qn = q
        if q[3] > 1.0:
            qn = Quaternion.normalize(q)

        angle = 2.0 * np.arccos(qn[3])
        s = np.sqrt(1.0 - qn[3]*qn[3])

        if s < 0.001:
            x = qn[0]
            y = qn[1]
            z = qn[2]
        else:
            x = qn[0]/s
            y = qn[1]/s
            z = qn[2]/s

        return [x, y, z, angle]

    @staticmethod
    def axisAngle2Quat( aa ):
        """Transform a axis angle orientation into a quaternion

        See https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
        for details.

        The function normalizes the axis angle representation.

        :param aa:
            An array with the axis angle transformation [rx, ry, rz, angle].
            The angle is in radians
        :return:
            An array with the quaternion components as [qx, qy, qz, qw]
        """

        # Normalization of the axis angle description
        angle = aa[3]
        aa = np.array(aa[0:3]).astype(float)
        l = np.linalg.norm(aa)
        l_float = l.astype('float32')

        a1 = float(aa[0]) / l_float
        a2 = aa[1] / l_float
        a3 = aa[2] / l_float

        # normalized aa
        aa = np.array([a1, a2, a3, float(angle)])


        # computing the quaternion
        s = np.sin(aa[3] / 2)
        x = aa[0] * s
        y = aa[1] * s
        z = aa[2] * s
        w = np.cos(aa[3] / 2)

        return [x, y, z, w]

    @staticmethod
    def quat2AxisAngle3(q):
        """Transform a quaternion into an axis-angle representation with there components only

        This is a 3-components axis angle representation. The length of the axis represents
        the angle.

        :param q:
            A array with a quaternion as [qx, qy, qz, qw]
        :return:
            An axis angle representation in an array as [rx * ang, ry * ang, rz * ang], with
            ang, the angle.
        """
        # get axis angle
        aa = Quaternion.quat2AxisAngle(q)

        # normalize the axis
        l = np.sqrt(aa[0] ** 2 + aa[1] ** 2 + aa[2] ** 2)

        if l == 0.0:
            return [0,0,0,0]

        aa[0] = aa[0] / l
        aa[1] = aa[1] / l
        aa[2] = aa[2] / l

        # multiply each axis with the angle in aa[3]
        aa3 = [0,0,0]
        aa3[0] = aa[0] * aa[3]
        aa3[1] = aa[1] * aa[3]
        aa3[2] = aa[2] * aa[3]
        return aa3

    @staticmethod
    def axisAngle32Quat( aa3 ):
        """Transform a 3-componennt axis-angle orientation into a quaternion.

        The 3-component axis-angle transformation encodes the angle as the lenght of the axis.
        The function returns the quaternion for this description.

        :param aa3: an array with the three components of an axis-angle orientation,
            in an array as [rx * ang, ry * ang, rz * ang], with ang, the angle.
        :return:
            An array with the components of a quaternion [qx, qy, qz, qw]
        """
        # get the length
        l = np.sqrt(aa3[0] ** 2 + aa3[1] ** 2 + aa3[2] ** 2)

        # divide each component by its length
        aa = [0,0,0,l]
        aa[0] = aa3[0] / l
        aa[1] = aa3[1] / l
        aa[2] = aa3[2] / l

        return Quaternion.axisAngle2Quat(aa)


    @staticmethod
    def distance(q1, q2):
        """Calculate the rotation distance between two quaternions.

        The function calculates the rotation difference between two quaternions as
            angle = 2.0 * arccos(  q1 * inv(q2) )

        See
        http://www.boris-belousov.net/2016/12/01/quat-dist/
        for details.
        NOTE THAT THE WEBPAGE HAS THE SIGNS INCORRECT

        :param q1:
            A array with a quaternion as [qx, qy, qz, qw]
        :param q2:
            A array with a quaternion as [qx, qy, qz, qw]
        :return: [float]
            The rotation distance between the quaternions as float
        """

        q1 = Quaternion.normalize(q1)
        q2 = Quaternion.normalize(q2)

        q2_i = Quaternion.conjugate(q2)

        # Extract from the first component, the cos component.
        sum = np.abs( - q1[0] * q2_i[0] - q1[1] * q2_i[1] - q1[2] * q2_i[2] + q1[3] * q2_i[3] )
        angle = 2.0 * np.arccos(sum)

        return angle

    @staticmethod
    def NormalizeList(quat):
        """Normalize all quats in a list of tuples with numpy serialization.

        The function normalizes all quaternions in a list of tuples using serialization.
        It expects teh quaternions to come as [ [qx, qy, qz, qw], [qx, qy, qz, qw], .... , [qx, qy, qz, qw], N]
        as shape [N, 4]

        :param quat: list
            List with quaternions of shape [N, 4], with N, the number of quaeternions, and
            4, the quaternion components [qx, qy, qz, qw].
        :return: list
            List with normalized quaternions of shape [N, 4]. The quaternions return the same
            way as they come.
        """
        N_ = quat.shape[0]
        len_ = np.sqrt(np.sum(np.power(quat[:, :], 2), axis=1))
        norm_ = np.divide(quat[:, :], len_.reshape(N_, 1))

        return norm_