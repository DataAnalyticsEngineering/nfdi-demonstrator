import matplotlib
from matplotlib import pyplot as plt
import torch
from typing import Tuple, Optional, List
import shutil
import time
from utils import get_param_fields

#plt.rcParams["text.usetex"] = True if shutil.which('latex') else False
#matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


class ThermalWidget:
    def __init__(self, samples, model, dtype=torch.float64, show_colorbars=True, figsize=[9,8], dpi=120, device="cpu"):
        self.samples = samples
        self.model = model
        self.device = device
        self.dtype = dtype
        self.args = {"device": self.device, "dtype": self.dtype}
        self.fig, ax = plt.subplots(3, 2, figsize=[9,8], dpi=120, gridspec_kw={'height_ratios': [1, 10, 10]})
        field = torch.zeros(self.samples[0].shape[-2:])
        self.cax_temp0 = ax[1,0].imshow(field.T, origin="lower", cmap="jet")
        ax[1,0].set_title(r"Temperature fluct. (load case 1) $\tilde{\theta} \, \mathrm{[K]}$")
        self.cax_flux0 = ax[1,1].imshow(field.T, origin="lower", cmap="jet")
        ax[1,1].set_title(r"Heat flux magn. (load case 1) $||\boldsymbol{q}|| \, \mathrm{[W/m^2]}$")
        self.cax_temp1 = ax[2,0].imshow(field.T, origin="lower", cmap="jet")
        ax[2,0].set_title(r"Temperature fluct. (load case 2) $\tilde{\theta} \, \mathrm{[K]}$")
        self.cax_flux1 = ax[2,1].imshow(field.T, origin="lower", cmap="jet")
        ax[2,1].set_title(r"Heat flux magn. (load case 2) $||\boldsymbol{q}|| \, \mathrm{[W/m^2]}$")
        if show_colorbars:
            self.fig.colorbar(self.cax_temp0, ax=ax[1,0])
            self.fig.colorbar(self.cax_flux0, ax=ax[1,1])
            self.fig.colorbar(self.cax_temp1, ax=ax[2,0])
            self.fig.colorbar(self.cax_flux1, ax=ax[2,1])
        
        for ax_handle in ax[1:,:].ravel():
            ax_handle.axis("off")
        
        gs = ax[0, 1].get_gridspec()
        for ax in ax[0, :]:
            ax.remove()
        ax_kappa = self.fig.add_subplot(gs[0, :])
        self.fig.suptitle(r"Effective thermal conductivity $\bar{\boldsymbol{\kappa}} \, \mathrm{[W/m^2]}$")
        ax_kappa.spines['right'].set_color('none')
        ax_kappa.spines['left'].set_color('none')
        ax_kappa.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
        ax_kappa.spines['top'].set_color('none')
        ax_kappa.xaxis.set_ticks_position('bottom')
        ax_kappa.set_yticks([])
        min_param, max_param = 0.1, 1.0
        self.line_kappa0 = plt.Line2D((min_param, min_param), (-1.0,1.0), color='b')
        self.line_kappa1 = plt.Line2D((max_param, max_param), (-1.0,1.0), color='b')
        self.line_reuss = plt.Line2D((min_param, min_param), (-1.0,1.0), color='g')
        self.line_voigt = plt.Line2D((max_param, max_param), (-1.0,1.0), color='g')
        self.line_eig0 = plt.Line2D((min_param, min_param), (-1.0,1.0), color='r')
        self.line_eig1 = plt.Line2D((max_param, max_param), (-1.0,1.0), color='r')
        ax_kappa.add_line(self.line_kappa0)
        ax_kappa.add_line(self.line_kappa1)
        ax_kappa.add_line(self.line_reuss)
        ax_kappa.add_line(self.line_voigt)
        ax_kappa.add_line(self.line_eig0)
        ax_kappa.add_line(self.line_eig1)
        ax_kappa.set(xlim=(min_param - 0.1, max_param + 0.1), ylim=(-1, 1))
        ax_kappa.legend([self.line_kappa1, self.line_reuss, self.line_eig0, self.line_eig1, self.line_voigt, self.line_kappa0],
                   [r"$\kappa_1$", "Reuss bound", r"$\lambda_1(\bar{\boldsymbol{\kappa}})$", r"$\lambda_2(\bar{\boldsymbol{\kappa}})$", "Voigt bound", r"$\kappa_0$"],
                   ncol=6, mode="expand", borderaxespad=0., bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', fancybox=True)
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.tight_layout()

    def update(self, ms_id, kappa1, alpha, print_times=False):
        start = time.time()
        image = self.samples[ms_id]
        params = torch.tensor([1., kappa1]).reshape(2, 1)
        param_field = get_param_fields(image, params).to(**self.args).unsqueeze(0)
    
        alpha_rad = torch.deg2rad(torch.tensor(alpha))
        loading = torch.tensor([[torch.cos(alpha_rad), -torch.sin(alpha_rad)], [torch.sin(alpha_rad), torch.cos(alpha_rad)]], **self.args)
        prepro_time = (time.time() - start) * 1000.0
    
        start = time.time()
        with torch.inference_mode():
            field = self.model(param_field, loading)
            if self.device != "cpu":
                torch.cuda.synchronize()
        compute_time = (time.time() - start) * 1000.0
    
        start = time.time()
        with torch.inference_mode():
            vol_frac = image.mean()
            reuss = 1. / (vol_frac.item() / params[1].item() + (1. - vol_frac.item()) / params[0].item())
            voigt = vol_frac.item() * params[1].item() + (1. - vol_frac.item()) * params[0].item()
            temp = field[...,0,:,:].detach().cpu()
            flux = field[...,1:,:,:]
            flux_norm = flux.norm(dim=-3).detach().cpu()
            hom_flux = flux.mean([-1,-2]).squeeze()
            kappa_bar = -hom_flux @ loading.inverse()
            eig_kappa = torch.linalg.eigvals(kappa_bar).real.cpu()
        postpro_time = (time.time() - start) * 1000.0
        
        start = time.time()
        self.line_kappa0.set_xdata([1.0, 1.0])
        self.line_kappa1.set_xdata([kappa1, kappa1])
        self.line_reuss.set_xdata([reuss, reuss])
        self.line_voigt.set_xdata([voigt, voigt])
        self.line_eig0.set_xdata([eig_kappa[0], eig_kappa[0]])
        self.line_eig1.set_xdata([eig_kappa[1], eig_kappa[1]])
        self.cax_temp0.set_data(temp[0,0].T)
        self.cax_flux0.set_data(flux_norm[0,0].T)
        self.cax_temp1.set_data(temp[0,1].T)
        self.cax_flux1.set_data(flux_norm[0,1].T)
        self.cax_temp0.autoscale()
        self.cax_flux0.autoscale()
        self.cax_temp1.autoscale()
        self.cax_flux1.autoscale()
        self.fig.canvas.draw()
        plot_time = (time.time() - start) * 1000.0
        if print_times:
            print(f"Times: preprocessing {prepro_time:.4f}ms, simulation {compute_time:.4f}ms, postprocessing {postpro_time:.4f}ms, plotting {plot_time:.4f}ms")


def plot_channel(
    field_ref: torch.Tensor,
    field_pred: torch.Tensor,
    channel: int,
    label_ref="",
    label_pred="",
    label_err="",
    ax: Optional[List[matplotlib.axes.Axes]] = None,
    cmap: str = "jet",
    centered=False,
    plot_error: bool = True,
    cmap_err: str = "seismic",
    cbar_label: Optional[str] = None,
    norm=None,
):
    """

    :param field_ref:
    :param field_pred:
    :param channel:
    :param ax:
    :param cmap:
    :param cmap_err:
    :param cbar_label:
    :return:
    """
    assert field_ref.ndim == 3
    assert field_pred.ndim == 3
    field_ref = field_ref.detach().cpu()
    field_pred = field_pred.detach().cpu()
    field_err = field_ref - field_pred

    standalone = True if ax is None else False
    if standalone:
        fig, ax = plt.subplots(1, 3 if plot_error else 2, figsize=[8, 2])
    for ax_handle in ax.ravel():
        ax_handle.axis("off")
    
    field_min, field_max = get_bounds(field_ref[channel], field_pred[channel], centered=centered)
    im_args = {"interpolation": "none", "origin": "lower", "extent": (-0.5, 0.5, -0.5, 0.5)}
    im_args_err = {**im_args, "norm": matplotlib.colors.CenteredNorm()}
    if norm is None:
        im_args["vmin"] = field_min
        im_args["vmax"] = field_max
    else:
        im_args["norm"] = norm
    
    im = ax[0].imshow(field_ref[channel].T, cmap=cmap, **im_args)
    pcm = plt.colorbar(im, ax=ax[0])
    if cbar_label is not None:
        pcm.ax.set_title(cbar_label)
    ax[0].set_title(label_ref)

    im = ax[1].imshow(field_pred[channel].T, cmap=cmap, **im_args)
    pcm = plt.colorbar(im, ax=ax[1])
    if cbar_label is not None:
        pcm.ax.set_title(cbar_label)
    ax[1].set_title(label_pred)

    if plot_error:
        im = ax[2].imshow(field_err[channel].T, cmap=cmap_err, **im_args_err)
        pcm = plt.colorbar(im, ax=ax[2])
        ax[2].set_title(label_err)

    if standalone:
        plt.tight_layout()
        plt.show()


def get_bounds(field_ref: torch.Tensor, field_pred: torch.Tensor, centered: bool = False) -> Tuple[float, float]:
    """
    Get minimum and maximum bound for plotting

    :param field_ref:
    :param field_pred:
    :param centered:
    :return: field_min, field_max
    """
    values_ref = torch.masked_select(field_ref, torch.logical_not(field_ref.isnan()))
    values_pred = torch.masked_select(field_pred, torch.logical_not(field_pred.isnan()))
    field_min = min(values_ref.min().item(), values_pred.min().item())
    field_max = max(values_ref.max().item(), values_pred.max().item())
    abs_range = max(abs(field_min), abs(field_max))
    if centered:
        return -abs_range, abs_range
    else:
        return field_min, field_max



def plot_kappa_bounds(kappa, kappa_lower, kappa_upper, params, file=None):
    """

    :return:
    """
    min_param, max_param = params.min().item(), params.max().item()

    fig, ax = plt.subplots(figsize=[6.3, 0.2], dpi=300)

    # Set the axis limits
    ax.set_xlim(min_param - 0.1, max_param + 0.1)
    # Add the ticks
    ax.set_xticks(torch.arange(min_param, max_param + 0.1, 0.1))

    # Remove y-axis
    ax.get_yaxis().set_visible(False)
    # Center x-axis
    # ax.set_aspect('equal')

    # add arrows
    ax.arrow(min_param, 0, max_param - min_param, 0, head_length=0., fc='k', ec='k')
    ax.arrow(max_param, 0, min_param - max_param, 0, head_length=0., fc='k', ec='k')

    # plot trivial bounds (material parameters of phases)
    trans = ax.get_xaxis_transform()
    ax.axvline(x=params[0], color='b')
    ax.text(params[0] - 0.01, 1.5, r"$\kappa_0$", transform=trans)
    ax.axvline(x=params[1], color='b')
    ax.text(params[1] - 0.01, 1.5, r"$\kappa_1$", transform=trans)

    # plot trivial bounds (material parameters of phases)
    trans = ax.get_xaxis_transform()
    ax.axvline(x=params[0])
    ax.text(params[0] - 0.01, 1.5, r"$\kappa_0$", transform=trans)
    ax.axvline(x=params[1])
    ax.text(params[1] - 0.01, 1.5, r"$\kappa_1$", transform=trans)

    # plot voigt reuss bounds
    trans = ax.get_xaxis_transform()
    ax.axvline(x=kappa_lower, color='g')
    ax.text(kappa_lower - 0.02, 1.5, r"$\kappa_\mathrm{lb}$", transform=trans)
    ax.axvline(x=kappa_upper, color='g')
    ax.text(kappa_upper - 0.02, 1.5, r"$\kappa_\mathrm{ub}$", transform=trans)

    # plot actual kappa eigenvalues
    eig_kappa = torch.linalg.eigvals(kappa).real
    ax.scatter(x=eig_kappa, y=torch.zeros_like(eig_kappa), marker="|", color="red")
    ax.text(eig_kappa.mean() - 0.03, 1.5, r"$\mathrm{eig}(\boldsymbol{\bar{\kappa}})$", transform=trans)

    # remove border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Show the plot
    from matplotlib.transforms import Bbox
    bbox = fig.get_tightbbox()
    bbox = Bbox([[bbox.x0, bbox.y0 - 0.1], [bbox.x1, bbox.y1 + 0.1]])

    if file is not None:
        plt.savefig(file, dpi=300, bbox_inches=bbox)
    plt.show()
