from ...frontend.base_frontend import ScatteringBase

from ..filter_bank import filter_bank
from ..utils import compute_padding
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec

class ScatteringBase2D(ScatteringBase):
    def __init__(self, J, shape, L=8, max_order=2, pre_pad=False,
            backend=None, out_type='array'):
        super(ScatteringBase2D, self).__init__()
        self.pre_pad = pre_pad
        self.L = L
        self.backend = backend
        self.J = J
        self.shape = shape
        self.max_order = max_order
        self.out_type = out_type

    def build(self):
        M, N = self.shape

        if 2 ** self.J > M or 2 ** self.J > N:
            raise RuntimeError('The smallest dimension should be larger than 2^J.')
        self._M_padded, self._N_padded = compute_padding(M, N, self.J)
        # pads equally on a given side if the amount of padding to add is an even number of pixels, otherwise it adds an extra pixel
        if not self.pre_pad:
            self.pad = self.backend.Pad([(self._M_padded - M) // 2, (self._M_padded - M+1) // 2, (self._N_padded - N) // 2,
                                (self._N_padded - N + 1) // 2], [M, N])
        else:
            self.pad = lambda x: x

        self.unpad = self.backend.unpad

    def create_filters(self):
        filters = filter_bank(self._M_padded, self._N_padded, self.J, self.L)
        self.phi, self.psi = filters['phi'], filters['psi']

    def scattering(self, x):
        """ This function should call the functional scattering."""
        raise NotImplementedError
    def plot_mst(self, x=None, scat_coeffs=None, src_img=None, label=None, cmap0='gray', cmap1='gray', cmap2='gray', log_r=False, r_is_scale=True, two_pi=False):
        '''
        Authors: Dr. Michael Glinsky and Francis Ogoke

        Plots the first and second order transforms and prints out the 0th order transform, and
        optionally the original image if it is given

        Parameters
        ----------
        x : The input data array for scattering, as would be done with self.scattering(x). Either x or
            scat_coeffs must be provided.
        scat_coeffs : numpy 3D array with indexes (MST_coeff_idx, MST_patch_i, MST_patch_j)
            Coefficients of the MST.
        src_img : numpy 2D array, optional
            image corresponding to the scat_coeffs. The default is None is plotted.
        label : str, optional
            label to title the plot. The default is None.
        cmap0 : str, optional
            colorbar to use for the original image. The default is 'gray'.
        cmap1 : str, optional
            colorbar to use for the 1st order MST. The default is 'gray'.
        cmap2 : str, optional
            colorbar to use for the 2nd order MST. The default is 'gray'.
        log_r : bool, optional
            plot a plot the radius logorithmically, otherwise linearly. The default is False.
        r_is_scale : bool, optional
            the radius is scale, otherwise it is 1/scale. The default is True.
        two_pi : bool, optional
            plot the second order sector from 0 to 2*pi, otherwise plot from 0 to pi. The default is False.

        Internal attributes used:
        J : int
            log2 of the samples in the patch, if J=log2(image) then there will be only one patch.
        L : int
            log2 of the number of angular sectors of the first order MST.
        Raises
        ------
        Exception
            If the dimensions of scat_coeffs do not match J and L.

        Returns
        -------
        fig : matplotlib figure object
            figure that was plotted of the MST.

        '''
        J = self.J
        L = self.L
        if not x and not scat_coeffs:
            raise ValueError("Either x or scat_coeffs must be provided.")
        scat_coeffs = self.scattering(x) if not scat_coeffs else scat_coeffs
        FONT_SIZE = 16

        # add option to plot colorbars
        print("coeffs shape: ", scat_coeffs.shape)

        len_order_0 = 1
        len_order_1 = J*L
        len_order_2 = (J*(J-1)//2)*(L**2)
        window_rows, window_columns = scat_coeffs.shape[1:]
        print("number of (order 0, order 1, order 2) coefficients: ", (len_order_0, len_order_1, len_order_2))
        print("number of window rows and columns: ", (window_rows, window_columns))

        length_mst = len_order_0 + len_order_1 + len_order_2
        if not scat_coeffs.shape[0] == length_mst:
            raise Exception('The value of L=%s and J=%s imply length of MST=%s, but it is %s' % (str(L),str(J),str(length_mst),str(scat_coeffs.shape[0])))

        ####################################################################
        # We now retrieve zeroth, first-order and second-order coefficients for the display.
        scat_coeffs_order_0 = scat_coeffs[0,:,:]

        scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]
        norm_order_1 = mpl.colors.Normalize(scat_coeffs_order_1.min(), scat_coeffs_order_1.max(), clip=True)
        mapper_order_1 = cm.ScalarMappable(norm=norm_order_1, cmap=cmap1)
        # Mapper of coefficient amplitude to a grayscale color for visualisation.

        scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]
        norm_order_2 = mpl.colors.Normalize(scat_coeffs_order_2.min(), scat_coeffs_order_2.max(), clip=True)
        mapper_order_2 = cm.ScalarMappable(norm=norm_order_2, cmap=cmap2)
        # Mapper of coefficient amplitude to a grayscale color for visualisation.

        # print out values of zeroth order coefficient
        print('MST of order 0 = %s' % (str(scat_coeffs_order_0)))

        if not src_img is None:
            fig = plt.figure(figsize=(19.2, 6))
            spec = fig.add_gridspec(ncols=3, nrows=1)
            gs = gridspec.GridSpec(1, 3, wspace=0.1)
            gs_order_1 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1])
            gs_order_2 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[2])
        else:
            fig = plt.figure(figsize=(12.8, 6))
            spec = fig.add_gridspec(ncols=2, nrows=1)
            gs = gridspec.GridSpec(1, 2, wspace=0.1)
            gs_order_1 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[0])
            gs_order_2 = gridspec.GridSpecFromSubplotSpec(window_rows, window_columns, subplot_spec=gs[1])

        # Start by plotting input
        if not src_img is None:
            ax = plt.subplot(gs[0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(src_img,cmap=cmap0,interpolation='nearest', aspect='auto')
            ax.axis('off')
            if label:
                ax.set_title(label, fontsize=FONT_SIZE)

        # Plot first-order scattering coefficients
        if not src_img is None:
            ax = plt.subplot(gs[1])
        else:
            ax = plt.subplot(gs[0])
        ax.set_xticks([])
        ax.set_yticks([])
        if not label is None:
            ax.set_title('MST_1( ' + label + ' )', fontsize=FONT_SIZE)

        l_offset = int(L - L / 2 - 1)  # follow same ordering as Kymatio for angles

        for row in range(window_rows):
            for column in range(window_columns):
                ax = fig.add_subplot(gs_order_1[row, column], projection='polar')
                ax.axis('off')
                coefficients = scat_coeffs_order_1[:, row, column]
                heights = []
                bottoms = []
                angles = []
                coeff_colors = []
                for j in range(J):
                    if r_is_scale:
                        if log_r:
                            height = 1
                            bottom = j
                        else:
                            if j == 0:
                                height = 4
                                bottom = 0
                            else:
                                height = 2 ** (j + 1)
                                bottom = height
                    else:
                        if log_r:
                            height = 1
                            bottom = J - 1 - j
                        else:
                            if j == J-1:
                                height = 2 ** (- J + 1.0)
                                bottom = 0
                            else:
                                height = 2 ** (- j - 1)
                                bottom = height
                    # print('plotting sector = ', j, bottom, bottom + height)
                    for l in range(L):
                        heights.append(height)
                        bottoms.append(bottom)
                        coeff = coefficients[l + j * L]
                        color = mapper_order_1.to_rgba(coeff)
                        coeff_colors.append(color)
                        angle = (l_offset - l) * np.pi / L
                        angles.append(angle)
                ax.bar(x=angles,
                        height=heights,
                        width=np.pi / L,
                        bottom=bottoms,
                        edgecolor=coeff_colors,
                        color=coeff_colors)
                ax.bar(x=np.array(angles) + np.pi,
                        height=heights,
                        width=np.pi / L,
                        bottom=bottoms,
                        edgecolor=coeff_colors,
                        color=coeff_colors)

        # Plot second-order scattering coefficients
        if not src_img is None:
            ax = plt.subplot(gs[2])
        else:
            ax = plt.subplot(gs[1])
        ax.set_xticks([])
        ax.set_yticks([])
        if not label is None:
            ax.set_title('MST_2( ' + label + ' )', fontsize=FONT_SIZE)

        for row in range(window_rows):
            for column in range(window_columns):
                ax = fig.add_subplot(gs_order_2[row, column], projection='polar')
                ax.axis('off')
                coefficients = scat_coeffs_order_2[:, row, column]
                heights = []
                bottoms = []
                angles = []
                coeff_colors = []
                for j1 in range(J - 1):
                    for j2 in range(j1 + 1, J):
                        if r_is_scale:
                            if log_r:
                                height = 1.0 / j2
                                bottom = (j2-1) + j1 / j2
                            else:
                                if j1 == 0:
                                    height = 4
                                    bottom = 2 ** (j2 + 1)
                                else:
                                    height = 2 ** (j1 + 1)
                                    bottom = 2 ** (j2 + 1) + height
                        else:
                            if log_r:
                                height = 1.0 / (J - j1 - 1)
                                bottom = (J - j1 - 2) + (J - j2 - 1) / (J - j1 - 1)
                            else:
                                if j2 == J-1:
                                    height = 2 ** (- J + 1.0)
                                    bottom = 2 ** (- j1 - 1)
                                else:
                                    height = 2 ** (- j2 - 1)
                                    bottom = 2 ** (- j1 - 1) + height
                        # print('plotting sector = ', j1, j2, bottom, bottom + height)
                        for l1 in range(L):
                            for l2 in range(L):
                                heights.append(height)
                                bottoms.append(bottom)
                                coeff_index = l1 * L * (J - j1 - 1) + l2 + L * (j2 - j1 - 1) + (L ** 2) * \
                                              (j1 * (J - 1) - j1 * (j1 - 1) // 2)
                                # indexing a bit complex which follows the order used by Kymatio to compute
                                # scattering coefficients
                                coeff = coefficients[coeff_index]
                                coeff_colors.append(mapper_order_2.to_rgba(coeff))
                                # split along angles first-order quadrants in L quadrants, using same ordering
                                # as Kymatio (clockwise) and center (with the 0.5 offset)

                                # equal split along radius is performed through height variable
                                if two_pi:
                                    angles.append((l_offset - l1) * np.pi / L + 0.5 * (L // 2 - l2 - 0.5) * np.pi / (L ** 2))
                                    width= 0.5 * np.pi / L ** 2,
                                    if l2 < L // 2:
                                        angles.append((l_offset - l1) * np.pi / L + 0.5 * (L // 2 + l2 + 0.5) * np.pi / (L ** 2))
                                        heights.append(height)
                                        bottoms.append(bottom)
                                    else:
                                        angles.append((l_offset - l1 - 1) * np.pi / L + 0.5 * (L // 2 + l2 + 0.5) * np.pi / (L ** 2))
                                        heights.append(height)
                                        bottoms.append(bottom)
                                else:
                                    angles.append((l_offset - l1) * np.pi / L + (L // 2 - l2 - 0.5) * np.pi / (L ** 2))
                                    width= np.pi / L ** 2,
                ax.bar(x=angles,
                        height=heights,
                        width=width,
                        bottom=bottoms,
                        edgecolor=coeff_colors,
                        color=coeff_colors)
                ax.bar(x=np.array(angles) + np.pi,
                        height=heights,
                        width=width,
                        bottom=bottoms,
                        edgecolor=coeff_colors,
                        color=coeff_colors)
        # return fig
    @property
    def M(self):
        warn("The attribute M is deprecated and will be removed in v0.4. "
        "Replace by shape[0].", DeprecationWarning)
        return int(self.shape[0])

    @property
    def N(self):
        warn("The attribute N is deprecated and will be removed in v0.4. "
        "Replace by shape[1].", DeprecationWarning)
        return int(self.shape[1])

    _doc_shape = 'M, N'

    _doc_instantiation_shape = {True: 'S = Scattering2D(J, (M, N))',
                                False: 'S = Scattering2D(J)'}

    _doc_param_shape = \
    r"""shape : tuple of ints
            Spatial support (M, N) of the input
        """

    _doc_attrs_shape = \
    r"""Psi : dictionary
            Contains the wavelets filters at all resolutions. See
            `filter_bank.filter_bank` for an exact description.
        Phi : dictionary
            Contains the low-pass filters at all resolutions. See
            `filter_bank.filter_bank` for an exact description.
        M_padded, N_padded : int
             Spatial support of the padded input.
        """

    _doc_param_out_type = \
    r"""out_type : str, optional
            The format of the output of a scattering transform. If set to
            `'list'`, then the output is a list containing each individual
            scattering path with meta information. Otherwise, if set to
            `'array'`, the output is a large array containing the
            concatenation of all scattering coefficients. Defaults to
            `'array'`.
        """

    _doc_attr_out_type = \
    r"""out_type : str
            The format of the scattering output. See documentation for
            `out_type` parameter above and the documentation for `scattering`.
        """

    _doc_class = \
    r"""The 2D scattering transform

        The scattering transform computes two wavelet transform
        followed by modulus non-linearity. It can be summarized as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_J^{{(0)}} x = x \star \phi_J$,

            $S_J^{{(1)}} x = [|x \star \psi^{{(1)}}_\lambda| \star \phi_J]_\lambda$, and

            $S_J^{{(2)}} x = [||x \star \psi^{{(1)}}_\lambda| \star
            \psi^{{(2)}}_\mu| \star \phi_J]_{{\lambda, \mu}}$.

        where $\star$ denotes the convolution (in space), $\phi_J$ is a
        lowpass filter, $\psi^{{(1)}}_\lambda$ is a family of bandpass filters
        and $\psi^{{(2)}}_\mu$ is another family of bandpass filters. Only
        Morlet filters are used in this implementation. Convolutions are
        efficiently performed in the Fourier domain.
        {frontend_paragraph}
        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J = 3
            M, N = 32, 32

            # Generate a sample signal.
            x = {sample}

            # Define a Scattering2D object.
            {instantiation}

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}(x)

        Parameters
        ----------
        J : int
            Log-2 of the scattering scale.
        {param_shape}L : int, optional
            Number of angles used for the wavelet transform. Defaults to `8`.
        max_order : int, optional
            The maximum order of scattering coefficients to compute. Must be
            either `1` or `2`. Defaults to `2`.
        pre_pad : boolean, optional
            Controls the padding: if set to False, a symmetric padding is
            applied on the signal. If set to True, the software will assume
            the signal was padded externally. Defaults to `False`.
        backend : object, optional
            Controls the backend which is combined with the frontend.
        {param_out_type}
        Attributes
        ----------
        J : int
            Log-2 of the scattering scale.
        {param_shape}L : int, optional
            Number of angles used for the wavelet transform.
        max_order : int, optional
            The maximum order of scattering coefficients to compute.
            Must be either `1` or `2`.
        pre_pad : boolean
            Controls the padding: if set to False, a symmetric padding is
            applied on the signal. If set to True, the software will assume
            the signal was padded externally.
        {attrs_shape}{attr_out_type}
        Notes
        -----
        The design of the filters is optimized for the value `L = 8`.

        The `pre_pad` flag is particularly useful when cropping bigger images
        because this does not introduce border effects inherent to padding.
        """

    _doc_scattering = \
    """Apply the scattering transform

       Parameters
       ----------
       input : {array}
           An input `{array}` of size `(B, M, N)`.

       Raises
       ------
       RuntimeError
           In the event that the input does not have at least two dimensions,
           or the tensor is not contiguous, or the tensor is not of the
           correct spatial size, padded or not.
       TypeError
           In the event that the input is not of type `{array}`.

       Returns
       -------
       S : {array}
           Scattering transform of the input. If `out_type` is set to
           `'array'` (or if it is not availabel for this frontend), this is
           a{n} `{array}` of shape `(B, C, M1, N1)` where `M1 = M // 2 ** J`
           and `N1 = N // 2 ** J`. The `C` is the number of scattering
           channels calculated. If `out_type` is `'list'`, the output is a
           list of dictionaries, with each dictionary corresponding to a
           scattering coefficient and its meta information. The actual
           coefficient is contained in the `'coef'` key, while other keys hold
           additional information, such as `'j'` (the scale of the filter
           used), and `'theta'` (the angle index of the filter used).
    """


    @classmethod
    def _document(cls):
        instantiation = cls._doc_instantiation_shape[cls._doc_has_shape]
        param_shape = cls._doc_param_shape if cls._doc_has_shape else ''
        attrs_shape = cls._doc_attrs_shape if cls._doc_has_shape else ''

        param_out_type = cls._doc_param_out_type if cls._doc_has_out_type else ''
        attr_out_type = cls._doc_attr_out_type if cls._doc_has_out_type else ''

        cls.__doc__ = ScatteringBase2D._doc_class.format(
            array=cls._doc_array,
            frontend_paragraph=cls._doc_frontend_paragraph,
            alias_name=cls._doc_alias_name,
            alias_call=cls._doc_alias_call,
            instantiation=instantiation,
            param_shape=param_shape,
            attrs_shape=attrs_shape,
            param_out_type=param_out_type,
            attr_out_type=attr_out_type,
            sample=cls._doc_sample.format(shape=cls._doc_shape))

        cls.scattering.__doc__ = ScatteringBase2D._doc_scattering.format(
            array=cls._doc_array,
            n=cls._doc_array_n)


__all__ = ['ScatteringBase2D']
