import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.fft import fft, fftfreq

def analyze_symmetry_slice(lattice, z_index=None, n_folds_expected=None):
    """
    Analizza una fetta 2D del cristallo per verificare matematicamente la simmetria.
    Usa la FFT sul profilo radiale per trovare le frequenze dominanti.
    """
    phi = lattice.phi
    if z_index is None:
        z_index = phi.shape[2] // 2 if phi.ndim == 3 else 0
        
    slice_img = phi[:, :, z_index] if phi.ndim == 3 else phi
    
    # 1. Trova il contorno dell'interfaccia (phi = 0.5)
    contours = measure.find_contours(slice_img, 0.5)
    if not contours:
        print("[Diagnostics] No crystal found in this slice.")
        return
    
    # Prendiamo il contorno più lungo (il cristallo principale)
    contour = max(contours, key=len)
    
    # 2. Calcola Centroide
    cy, cx = np.mean(contour, axis=0)
    
    # 3. Coordinate Polari (rispetto al centroide)
    # contour[:, 0] è Y, contour[:, 1] è X
    dy = contour[:, 0] - cy
    dx = contour[:, 1] - cx
    radii = np.sqrt(dx**2 + dy**2)
    angles = np.arctan2(dy, dx)
    
    # Ordiniamo per angolo per il plot
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    radii = radii[sort_idx]
    
    # 4. Analisi di Fourier (FFT) per rilevare n_folds
    # Interpoliamo su una griglia uniforme per la FFT
    angles_uniform = np.linspace(-np.pi, np.pi, 1024)
    radii_uniform = np.interp(angles_uniform, angles, radii)
    
    yf = fft(radii_uniform - np.mean(radii_uniform)) # Togliamo la componente DC (raggio medio)
    xf = fftfreq(1024, 1 / (2*np.pi)) # Frequenze angolari
    
    # Prendiamo solo la parte positiva dello spettro
    n_samples = 1024 // 2
    power = 2.0/1024 * np.abs(yf[0:n_samples])
    freqs = xf[0:n_samples] / (2*np.pi) # Normalizziamo a "folds"
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: La forma reale
    axes[0].imshow(slice_img, cmap='gray_r')
    axes[0].plot(contour[:, 1], contour[:, 0], 'r-', linewidth=2)
    axes[0].set_title(f"Cross Section (z={z_index})")
    
    # Plot 2: Raggio vs Angolo (Coordinate Polari srotolate)
    axes[1].plot(angles, radii)
    axes[1].set_xlabel("Angle (radians)")
    axes[1].set_ylabel("Radius (pixels)")
    axes[1].set_title("Radial Profile (Anisotropy Signature)")
    axes[1].grid(True)
    
    # Plot 3: Spettro delle frequenze (La prova schiacciante)
    axes[2].plot(freqs[:20], power[:20], 'o-') # Mostriamo solo le prime 20 armoniche
    axes[2].set_xlabel("Symmetry Mode (n-fold)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title("Symmetry Spectrum (FFT)")
    axes[2].set_xticks(range(0, 21))
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

def check_interface_stability(lattice):
    """
    Controlla se l'interfaccia è risolta numericamente in modo corretto.
    Un'interfaccia Phase-Field deve essere spalmata su 3-5 pixel.
    Se è di 1 pixel (gradino), la fisica è rotta (Numerical Pinning).
    """
    phi = lattice.phi
    # Prendiamo una linea che passa dal centro verso l'esterno
    mid_x, mid_y = phi.shape[0]//2, phi.shape[1]//2
    mid_z = phi.shape[2]//2 if phi.ndim == 3 else 0
    
    if phi.ndim == 3:
        profile = phi[mid_x, mid_y:, mid_z]
    else:
        profile = phi[mid_x, mid_y:]
        
    # Calcoliamo i gradienti lungo il profilo
    grad = np.abs(np.gradient(profile))
    max_grad = np.max(grad)
    interface_width_pixels = 1.0 / max_grad if max_grad > 0 else 0
    
    print(f"\n=== NUMERICAL STABILITY DIAGNOSTICS ===")
    print(f"Max Gradient: {max_grad:.4f}")
    print(f"Estimated Interface Width: {interface_width_pixels:.2f} pixels")
    
    if 2.5 <= interface_width_pixels <= 8.0:
        print("✅ STABILITY OK: Interface is well resolved.")
    elif interface_width_pixels < 2.5:
        print("❌ CRITICAL WARNING: Interface is too sharp (Pinning likely). Increase epsilon or reduce dx.")
    else:
        print("⚠️ WARNING: Interface is very diffuse. This is stable but might be computationally wasteful.")

    plt.figure()
    plt.plot(profile, 'b.-', label=r'$\phi$ profile')
    plt.title("Interface Profile Check")
    plt.xlabel("Distance from center (pixels)")
    plt.ylabel(r"Phase Field $\phi$")
    plt.grid()
    plt.legend()
    plt.show()

def verify_growth_law(lattice):
    """
    Verifica se la velocità di crescita è fisicamente coerente.
    In regime cinetico (Eden-like), il raggio deve crescere linearmente col tempo.
    """
    if np.all(lattice.history == -1):
        print("No history data available.")
        return

    max_epoch = int(np.max(lattice.history))
    times = np.arange(1, max_epoch + 1)
    equivalent_radii = []

    for t in times:
        # Volume (numero di voxel solidificati all'epoca <= t)
        vol = np.sum((lattice.history > -1) & (lattice.history <= t))
        
        # Raggio equivalente di una sfera con quel volume
        # V = 4/3 * pi * R^3  --> R = (3V / 4pi)^(1/3)
        r_eq = (3 * vol / (4 * np.pi))**(1/3)
        equivalent_radii.append(r_eq)

    # Fit lineare: R = v * t + c
    coeffs = np.polyfit(times, equivalent_radii, 1)
    velocity = coeffs[0]
    
    # Calcolo R^2 del fit per vedere se è veramente lineare
    p = np.poly1d(coeffs)
    r_pred = p(times)
    r2 = 1 - (np.sum((equivalent_radii - r_pred)**2) / np.sum((equivalent_radii - np.mean(equivalent_radii))**2))

    print(f"\n=== PHYSICAL GROWTH LAW DIAGNOSTICS ===")
    print(f"Growth Regime: Kinetic/Eden-like")
    print(f"Average Radial Velocity: {velocity:.4f} pixels/epoch")
    print(f"Linearity (R^2): {r2:.4f}")
    
    if r2 > 0.98:
        print("✅ PHYSICS OK: Growth is linear as expected for kinetic regime.")
    else:
        print("⚠️ ANOMALY: Growth is not linear. Check for boundary effects or instability.")

    plt.figure()
    plt.plot(times, equivalent_radii, label="Simulation Data")
    plt.plot(times, r_pred, 'r--', label=f"Linear Fit (v={velocity:.3f})")
    plt.xlabel("Time (Epoch)")
    plt.ylabel("Equivalent Radius")
    plt.title("Growth Law Verification")
    plt.legend()
    plt.grid()
    plt.show()