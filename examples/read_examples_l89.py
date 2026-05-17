"""read_examples_l89 - Example usage script for reading Landsat 8/9 data"""

import os
import time
import configparser

import matplotlib

matplotlib.use("Agg")  # Non-GUI backend

from eoio import read


__author__ = "Maddie Stedman <maddie.stedman@npl.co.uk>"

__all__ = []

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "eoio",
    "etc",
    "test_file_paths.config",
)
config = configparser.ConfigParser()
config.read(CONFIG_PATH)
# Assume config has a section [Landsat] and key 'l8_dir'
example_data_path = config.get("Landsat", "l8_dir", fallback=None)

if __name__ == "__main__":
    # Check if we have any Landsat files
    if len(example_data_path) == 0:
        print("No Landsat 8 or 9 L1 files found. Please check the data directory.")
        exit(1)

    # Use the first available Landsat file
    landsat_file = example_data_path
    print(f"Reading Landsat file: {landsat_file}")

    # Read Landsat data with selected bands (B2=Blue, B3=Green, B4=Red, B5=NIR) and angles
    start_time = time.time()
    ds = read(
        landsat_file,
        vars_sel={
            "meas": ["B2", "B3", "B4", "B10"],  # Read only a few bands for efficiency
            "aux": [
                "solar_zenith_angle",
                "solar_azimuth_angle",
                "viewing_zenith_angle",
                "viewing_azimuth_angle",
            ],
        },
        read_params={
            "use_chunks": True,
            "metadata_level": "all",
            "save_extracted": True,
        },
        processors={
            "units.convert": {
                "to": {"vnir": "radiance", "tir": "brightness_temperature"},
                "var_names": ["B2", "B3", "B4", "B10"],  #'all'
            }
        },
    )
    end_time = time.time()
    print(f"Read time: {end_time - start_time:.2f} seconds")

    for attr in ds.attrs:
        print(f"{attr}: {ds.attrs[attr]}")

    for var in ds.data_vars:
        print(f"\nVariable: {var}")
        for attr in ds[var].attrs:
            print(f"  {attr}: {ds[var].attrs[attr]}")

    # # Display dataset information
    # print("\n" + "=" * 60)
    # print("Dataset Overview:")
    # print("=" * 60)
    # print(ds)

    # # Display band information
    # print("\n" + "=" * 60)
    # print("Band Information:")
    # print("=" * 60)
    # for band in ["B2", "B3", "B4", "B10"]:
    #     if band in ds:
    #         print(f"{band} shape: {ds[band].shape}")
    #         print(
    #             f"  - wavelength: {ds[band].attrs.get('band_central_wavelength', 'N/A')} nm"
    #         )
    #         print(
    #             f"  - resolution: {ds[band].attrs.get('spatial_resolution', 'N/A')} m"
    #         )

    # # Display angle information
    # print("\n" + "=" * 60)
    # print("Angle Information:")
    # print("=" * 60)
    # angle_vars = [
    #     "solar_zenith_angle",
    #     "solar_azimuth_angle",
    #     "viewing_zenith_angle",
    #     "viewing_azimuth_angle",
    # ]
    # for angle in angle_vars:
    #     if angle in ds:
    #         print(
    #             f"{angle}: shape={ds[angle].shape}, "
    #             f"min={float(ds[angle].min()):.2f}°, "
    #             f"max={float(ds[angle].max()):.2f}°"
    #         )

    # # Create plots
    # print("\n" + "=" * 60)
    # print("Creating plots...")
    # print("=" * 60)

    # # Plot RGB bands
    # fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # # Blue band (B2)
    # ds.B2.plot(ax=axes[0, 0], robust=True, cmap="Blues")
    # axes[0, 0].set_title("Band 2 - Blue (0.45-0.51 µm)")

    # # Green band (B3)
    # ds.B3.plot(ax=axes[0, 1], robust=True, cmap="Greens")
    # axes[0, 1].set_title("Band 3 - Green (0.53-0.59 µm)")

    # # Red band (B4)
    # ds.B4.plot(ax=axes[1, 0], robust=True, cmap="Reds")
    # axes[1, 0].set_title("Band 4 - Red (0.64-0.67 µm)")

    # # NIR band (B10)
    # ds.B10.plot(ax=axes[1, 1], robust=True, cmap="RdYlGn")
    # axes[1, 1].set_title("Band 10 - Thermal (0.10-0.12 µm)")

    # plt.tight_layout()
    # plt.savefig("landsat_bands_plot.png", dpi=150)
    # print("Saved: landsat_bands_plot.png")
    # plt.close()

    # # Plot angles
    # fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # # Solar zenith angle
    # ds.solar_zenith_angle.plot(ax=axes[0, 0], cmap="viridis")
    # axes[0, 0].set_title("Solar Zenith Angle")

    # # Solar azimuth angle
    # ds.solar_azimuth_angle.plot(ax=axes[0, 1], cmap="twilight")
    # axes[0, 1].set_title("Solar Azimuth Angle")

    # # Viewing zenith angle
    # ds.viewing_zenith_angle.plot(ax=axes[1, 0], cmap="viridis")
    # axes[1, 0].set_title("Viewing Zenith Angle")

    # # Viewing azimuth angle
    # ds.viewing_azimuth_angle.plot(ax=axes[1, 1], cmap="twilight")
    # axes[1, 1].set_title("Viewing Azimuth Angle")

    # plt.tight_layout()
    # plt.savefig("landsat_angles_plot.png", dpi=150)
    # print("Saved: landsat_angles_plot.png")
    # plt.close()

    # # Create an RGB composite
    # fig, ax = plt.subplots(figsize=(10, 10))

    # # Normalize bands to 0-1 range for RGB visualization
    # red = (ds.B4 - ds.B4.min()) / (ds.B4.max() - ds.B4.min())
    # green = (ds.B3 - ds.B3.min()) / (ds.B3.max() - ds.B3.min())
    # blue = (ds.B2 - ds.B2.min()) / (ds.B2.max() - ds.B2.min())

    # # Stack into RGB
    # import numpy as np

    # rgb = np.dstack([red.values, green.values, blue.values])

    # # Apply simple contrast stretch
    # rgb = np.clip(rgb * 2.5, 0, 1)

    # ax.imshow(rgb)
    # ax.set_title("Landsat 8 RGB Composite (B4, B3, B2)")
    # ax.axis("off")

    # plt.tight_layout()
    # plt.savefig("landsat_rgb_composite.png", dpi=150)
    # print("Saved: landsat_rgb_composite.png")
    # plt.close()

    # print("\nDone! All plots have been saved.")
