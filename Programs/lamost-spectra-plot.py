"""
Plotting LAMOST spectra for Galactic Sources 
Autor: Luis A. Gutiérrez Soto
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns
from astropy.io import fits
import argparse
import os

def plot_spectrum(wavelengths, flux, emission_lines, xlim=None, ylim=None, 
                  displace_lines=None, displace_lines_x=None, inset_positions=None,
                  inset_xlim=None, inset_ylim=None, 
                  inset_xlim2=None, inset_ylim2=None, 
                  savefig=None, title=None, object_name=None):
    """
    Plots spectrum with emission lines labels.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=20)
    ax.set_ylabel(r'Relative flux', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    
    # Plot the spectrum first
    ax.plot(wavelengths, flux, c="k", lw=1, zorder=3)
    
    # Set axis limits before other elements
    if xlim:
        ax.set_xlim(*xlim)
    
    # Add title and object name
    if title:
        ax.set_title(title, fontsize=18)
    if object_name:
        ax.text(0.7, 0.95, object_name, transform=ax.transAxes, 
                fontsize=16, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7))

    # Calculate max flux around emission lines for label positioning
    max_flux = []
    for line, wavelength in emission_lines.items():
        mask = (wavelengths >= wavelength - 10) & (wavelengths <= wavelength + 10)
        
        if np.any(mask):
            flux_values = flux[mask]
            max_flux.append(np.nanmax(flux_values))
        else:
            max_flux.append(0.1 * np.nanmax(flux))  # Fallback value

    # Plot labels with adjusted positions
    bbox_props = dict(boxstyle="round", fc="w", ec="0.6", alpha=0.7, pad=0.1)
    
    # Use a consistent light gray for all dashed lines
    line_color = "lightgray"
    
    for i, (label, wavelength) in enumerate(emission_lines.items()):
        # Skip lines outside current wavelength range
        if xlim and (wavelength < xlim[0] or wavelength > xlim[1]):
            continue
            
        # Default positions
        y_pos = max_flux[i]
        x_pos = wavelength
        
        # Modificación clave: Usar porcentaje del rango Y en lugar de valor absoluto
        if displace_lines and label in displace_lines:
            # Cambio a desplazamiento relativo
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_pos += displace_lines[label] * y_range / 100  # displace_lines ahora en % del rango Y
            
        # Apply X displacement if specified
        if displace_lines_x and label in displace_lines_x:
            x_pos += displace_lines_x[label]
        
        # Plot vertical DASHED line with consistent color
        ax.axvline(wavelength, color=line_color, lw=1.2, alpha=0.9, ls='--')
        
        # Plot annotation with BLACK text
        ax.annotate(
            label, 
            (x_pos, y_pos), 
            alpha=0.9, 
            size=12,
            xytext=(1.5, 5),  # Small vertical offset only
            textcoords='offset points', 
            ha='center', 
            va='bottom',
            rotation=90, 
            bbox=bbox_props,
            color="black"  # All labels in black
        )

    # First inset plot
    if inset_xlim and inset_ylim:
        axins = inset_axes(ax, width=2.5, height=2.5, loc='upper right', bbox_to_anchor=inset_positions)
        axins.plot(wavelengths, flux, c="k", lw=0.8)
        axins.set_xlim(*inset_xlim)
        axins.set_ylim(*inset_ylim)
        axins.tick_params(labelsize=10)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
        for label_, x in emission_lines.items():
            axins.annotate(label_, (x, max_flux[list(emission_lines.keys()).index(label_)]), alpha=1, size=14,
                                xytext=(7.5, 5.6), textcoords='offset points', ha='right', va='bottom',
                                rotation=90, bbox=bbox_props, zorder=200)
        axins.tick_params(axis='x', labelsize=14)
        axins.tick_params(axis='y', labelsize=14)

    # # Second inset plot
    # if inset_xlim2 and inset_ylim2:
    #     axins2 = inset_axes(ax, width=2.5, height=2.5, loc='lower right')
    #     axins2.plot(wavelengths, flux, c="k", lw=0.8)
    #     axins2.set_xlim(*inset_xlim2)
    #     axins2.set_ylim(*inset_ylim2)
    #     axins2.tick_params(labelsize=10)
    #     mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")

    if ylim:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    
    if savefig:
        plt.savefig(savefig, dpi=200, bbox_inches='tight')
    

def main():
    parser = argparse.ArgumentParser(description="Plot LAMOST spectra for Galactic sources")
    parser.add_argument("source", help="Source spectrum filename (without .fits extension)")
    parser.add_argument("--name", default="Object", help="Object name")
    parser.add_argument("--xmin", type=float, default=3700, help="Min wavelength")
    parser.add_argument("--xmax", type=float, default=9000, help="Max wavelength")
    parser.add_argument("--ymin", type=float, default=None, help="Min flux")
    parser.add_argument("--ymax", type=float, default=None, help="Max flux")
    parser.add_argument("--output", help="Output filename")
    
    args = parser.parse_args()

    # Set default output filename if not provided
    if not args.output:
        args.output = f"{args.source}.pdf"

    # Load FITS file
    file_path = f"{args.source}.fits"
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return

    with fits.open(file_path) as hdul:
        # Handle different LAMOST data formats
        if len(hdul) > 1 and 'FLUX' in hdul[1].columns.names:  # DR8+ format
            data = hdul[1].data
            wl = data['WAVELENGTH'].flatten()
            flux = data['FLUX'].flatten()
        elif len(hdul) > 0 and isinstance(hdul[0].data, np.ndarray):  # Older format
            data = hdul[0].data
            if data.shape[0] >= 3:
                wl = data[2]
                flux = data[0]
            else:
                print("Error: Unexpected data format in FITS file")
                return
        else:
            print("Error: Could not read spectrum from FITS file")
            return
    
    # Emission lines (rest frame) - Galactic sources
    emission_lines = {
    "H I 3835": 3835.39,
    "H I 3889": 3889.05,
    "Ca II 3934": 3933.66,
    "H I 3970": 3970.07,
    "Hδ": 4101.74,
    "Hγ": 4340.471,
    "He I 4472": 4471.50 ,
    "He II 4686": 4685.68,
    "Hβ": 4861.33,
    "He I 4922": 4921.93,
    "He I 5016": 5015.68,
    "He I 5876": 5875.624,
    "Hα": 6562.82,
    "He I 6678": 6678.16,
    "Raman O VI": 6830.0,  # <--- Pero en el espectro el pica esta en línea en 6831 Å
        "He I 7065":   7065.25,
        "O I 7774": 7774.17, # Tengo que verificar esta linea no estoy seguro que sea correcta
        "He I 8445": 8444.69, 
        "H I 8502": 8502.48,  
        "H I 8545": 8545.38,
        #"He I 8608": 8608.31,
        "H I 8665": 8665.02,
        "H I 8750": 8750.47,
        "H I 8863": 8862.78,
        "H I 9015": 9014.91,
}
   

    # Custom displacement factors (absolute values in data units)
    displace_lines = {
        "H I 3970": 8.,  # Move Hα label up by 0.5 flux units
        "Hα": -53,
        "H I 8502": 9.,
         #"H I 8665": 5.,
    }
    
    displace_lines_x = {
        "Hα": 50,  # Move label right by 5 Angstroms
    }

    # Handle ymin/ymax properly
    ymin = args.ymin if args.ymin is not None else np.nanmin(flux)
    ymax = args.ymax if args.ymax is not None else np.nanmax(flux)
    ylim = (ymin, ymax)

    # Precompute y-axis limits for insets
    mask1 = (wl > 4670) & (wl < 4705)
    mask2 = (wl > 6500) & (wl < 6800)
    
    y_inset1 = 1.2 * np.nanmax(flux[mask1]) if np.any(mask1) else 1.0
    y_inset2 = 1.2 * np.nanmax(flux[mask2]) if np.any(mask2) else 1.0

    # Plot the spectrum
    plot_spectrum(
        wl, 
        flux, 
        emission_lines,
        xlim=(args.xmin, args.xmax),
        ylim=ylim,
        displace_lines=displace_lines,
        displace_lines_x=displace_lines_x,
        inset_positions= (-10, 100, 550, 300),
        inset_xlim=(4670, 4705),
        inset_ylim=(20, y_inset1),
        inset_xlim2=(6500, 6800),
        inset_ylim2=(0, y_inset2),
        savefig=args.output,
        object_name=args.name
    )

if __name__ == "__main__":
    main()
