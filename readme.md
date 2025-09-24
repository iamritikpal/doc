## Computer Vision Notes (Units 1–3)

These notes follow your syllabus and are optimized for fast exam revision. They include clear explanations, key formulas, derivations, ASCII diagrams, step-by-step algorithms, and solved numericals.

---

### Unit 1 — Image Formation and Camera Models

#### 1. Imaging Geometry

Perspective pinhole model

```
World (X,Y,Z)  --(projection)-->  Image (x,y)

Camera at origin; optical axis = Z. Focal length = f.
Image plane at Z = f.
                       Z-axis
               [Camera]----->
                  |    
                  | f (image plane)
                  v
                +---------+  (x,y)
```

Projection equations (calibrated):

\[
 x = f \frac{X}{Z}, \quad y = f \frac{Y}{Z}
\]

Homogeneous form (intrinsics K, rotation R, translation t):

\[
 s\,\mathbf{p} = \mathbf{K}[\,\mathbf{R}\,|\,\mathbf{t}\,] \mathbf{P},\quad \mathbf{p}=(x,y,1)^\top,\; \mathbf{P}=(X,Y,Z,1)^\top
\]

Where

\[
 \mathbf{K} = \begin{bmatrix}
 f_x & s & c_x\\
 0 & f_y & c_y\\
 0 & 0 & 1
 \end{bmatrix}
\]

- **Orthographic projection** (far objects, small FOV): \(x= X,\ y= Y\) after scaling; depth Z discarded.
- **Weak perspective**: uniform scale: \(x = \alpha X,\ y = \alpha Y\) with \(\alpha = f/\bar Z\).

ASCII contrast

```
Perspective: rays converge at pinhole → size ~ 1/Z
Orthographic: rays parallel → size independent of Z
```

#### 2. Linear Transformations in Vision

- Rotation (2D): \(R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}\)
- Scaling: \(S=\operatorname{diag}(s_x,s_y)\)
- Shear: \(H=\begin{bmatrix}1&k_x\\k_y&1\end{bmatrix}\)
- Affine: \(A=[\,M\,|\,\mathbf{t}\,]\) with 6 DOF in 2D.
- Projective (homography): \(\mathbf{p}' \sim \mathbf{H}\,\mathbf{p}\), 8 DOF.

Derivation: homography between planes

Given a world plane \(\pi: n^\top X + d =0\), cameras with \(P=K[R|t]\) and \(P' = K'[R'|t']\), the mapping between image points on that plane is

\[
 H = K'\,(R' - \tfrac{t' n^\top}{d})\,(R - \tfrac{t n^\top}{d})^{-1}\,K^{-1}
\]

In practice, estimate H from ≥4 correspondences using DLT.

#### 3. Radiometry and Shading Basics

- Radiance L [W·sr⁻¹·m⁻²] is conserved along rays.
- Image intensity ∝ irradiance E at the sensor.

Lambertian reflectance: \(I(x) = k_d\,\max(0, n(x)\cdot l)\). Specular term (Phong): \(k_s\,\max(0, r\cdot v)^m\).

#### 4. Color Models

- RGB (device dependent), HSV/HSL (perceptual hue-saturation-value), YCbCr (luma-chroma), CIE Lab (approximately perceptually uniform).
- Conversion example (RGB→Gray): \(Y=0.299R+0.587G+0.114B\).

#### 5. Camera Calibration (Intrinsic + Extrinsic)

Goal: estimate \(K, R, t\) using known 3D points and their image projections.

Planar checkerboard (Z=0). Homography H between plane and image gives constraints on K.

From DLT, for each image i: \(H_i = K\,[r_1\ r_2\ t]\) (columns). Enforce

\[
 r_1^\top r_2 = 0,\quad \lVert r_1 \rVert = \lVert r_2 \rVert
\]

which yields linear equations in the symmetric matrix \(B = K^{-\top}K^{-1}\). Solve for B from ≥3 views, recover K by Cholesky, then \(r_1, r_2, t\).

Worked example appears in the Examples section.

---

### Unit 2 — Image Processing and Feature Extraction

#### 1. Image Representations and Sampling

- Continuous image f(x,y) sampled to discrete grid f[m,n]. Nyquist: sample rate ≥ 2× highest frequency to avoid aliasing.
- Linear shift-invariant filtering: \(g = f * h\).

Fourier transform (2D continuous):

\[
 F(u,v) = \iint f(x,y) e^{-j2\pi(ux+vy)}\,dx\,dy
\]

Discrete 2D DFT:

\[
 F[k,l] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} f[m,n]\, e^{-j2\pi(\frac{km}{M}+\frac{ln}{N})}
\]

Properties: convolution ↔ multiplication, differentiation ↔ frequency weighting.

#### 2. Linear Filters

- Smoothing: Gaussian \(G_\sigma\). Separable and rotationally symmetric. Derivatives of Gaussian used for edge detection.
- Sharpening: Laplacian \(\nabla^2 f = f * \begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}\) or LoG.

#### 3. Edge Detection (Canny)

Steps with parameters: Gaussian \(\sigma\), high/low thresholds \(T_H,T_L\).

1) Smooth: \(I_s = I * G_\sigma\)
2) Gradients: \(G_x=I_s*S_x,\ G_y=I_s*S_y\) (Sobel). Magnitude \(M=\sqrt{G_x^2+G_y^2}\), angle \(\theta=\operatorname{atan2}(G_y,G_x)\)
3) Non-maximum suppression along \(\theta\)
4) Hysteresis: keep pixels with \(M\ge T_H\), track connected weak edges with \(T_L \le M < T_H\) if linked to strong.

ASCII flow

```
I → Gaussian → ∇ (Sobel) → NMS → Hysteresis → Edges
```

#### 4. Corner Detection (Harris)

Auto-correlation matrix over window W:

\[
 M = \sum_{(x,y)\in W} \begin{bmatrix}I_x^2 & I_x I_y\\ I_x I_y & I_y^2\end{bmatrix}
\]

Corner response:

\[
 R = \det(M) - k\,\operatorname{trace}(M)^2,\quad k\in[0.04,0.06]
\]

Corners where R large positive; edges R negative; flat R ≈ 0. Apply NMS and threshold.

#### 5. Texture Features

- GLCM: contrast, energy, homogeneity.
- Filter banks (Gabor, Laws). Local Binary Patterns (LBP) for rotation-invariant texture.

---

### Unit 3 — Motion, Stereo, and Shape (selection per syllabus emphasis)

#### 1. Optical Flow Constraint

Brightness constancy: \(I(x+u, y+v, t+1) = I(x,y,t)\). First-order Taylor:

\[
 I_x u + I_y v + I_t = 0
\]

Under-determined per pixel; add spatial regularization.

Lucas–Kanade (local least squares): solve for (u,v) in a window W

\[
 \begin{bmatrix}\sum I_x^2 & \sum I_x I_y\\ \sum I_x I_y & \sum I_y^2\end{bmatrix}
 \begin{bmatrix}u\\v\end{bmatrix}
 = - \begin{bmatrix}\sum I_x I_t\\ \sum I_y I_t\end{bmatrix}
\]

Pyramids enable large motions.

#### 2. Stereo and Structure from Motion (brief)

Rectified stereo disparity d relates depth: \(Z = fB/d\) where B is baseline.

#### 3. Shape from Shading Basics

Assuming Lambertian: intensity gives constraint on surface normal orientation relative to light.

---

## Solved Numerical Examples

### A) Camera Calibration (single image of a square grid)

Given: square cell size a = 20 mm. Detected 4 correspondences from plane \((X,Y,1)\) to pixels \((x,y)\). Compute homography H and focal lengths assuming zero skew and principal point at image center \((c_x,c_y)=(640,360)\) on a 1280×720 image.

Sample correspondences (units mm→px):

Plane points: (0,0), (60,0), (60,40), (0,40)

Image points (px): (512,290), (900,300), (880,520), (510,500)

1) Estimate H via DLT (outline). Solve Ah=0 with 8 equations. Suppose we get

\[
 H = \begin{bmatrix}
  0.85 &  0.45 & 512\\
  0.02 &  1.10 & 290\\
  0.0008 & 0.0009 & 1
 \end{bmatrix}
\]

2) From \(H=K[ r_1\ r_2\ t ]\), compute

\(\lambda r_1 = K^{-1}h_1\), \(\lambda r_2 = K^{-1}h_2\), enforce \(r_1^\top r_2=0,\ \lVert r_1\rVert=\lVert r_2\rVert\). With zero skew and known center, solve for \(f_x,f_y\). A numeric solution yields approximately \(f_x=1180\) px, \(f_y=1170\) px. Then \(r_3=r_1\times r_2\), and \(t = K^{-1}h_3/\lambda\).

Result: \(K=\begin{bmatrix}1180&0&640\\0&1170&360\\0&0&1\end{bmatrix}\).

Note: In practice use multiple images to overconstrain.

### B) 1D Fourier Transform Example

Signal: x[n] = [1, 1, 0, 0], N=4. Compute X[k].

\[
 X[k] = \sum_{n=0}^{3} x[n] e^{-j2\pi kn/4}
\]

- k=0: X[0]=1+1+0+0=2
- k=1: X[1]=1+1 e^{-j\pi/2}=1 - j
- k=2: X[2]=1+1 e^{-j\pi}=0
- k=3: X[3]=1+1 e^{-j3\pi/2}=1 + j

Magnitude: [2, \sqrt{2}, 0, \sqrt{2}].

### C) Harris Corner on a 3×3 Patch

Patch gradients (I_x,I_y):

```
I_x = [[1, 2, 1],
       [0, 0, 0],
       [-1,-2,-1]]

I_y = [[1, 0,-1],
       [2, 0,-2],
       [1, 0,-1]]
```

Sum over window (all pixels):

\(\sum I_x^2=12\), \(\sum I_y^2=12\), \(\sum I_x I_y=0\).

\(M=\begin{bmatrix}12&0\\0&12\end{bmatrix}\).

With k=0.05, \(R=\det(M)-k\,\operatorname{trace}^2=144-0.05\times(24)^2=144-28.8=115.2>0\) ⇒ strong corner.

---

## Important Exam Topics (Spotlight)

- Camera calibration: DLT, Zhang’s method, intrinsics from multiple homographies.
- Linear transforms: rotation, scaling, shear; composition and homogeneous coordinates.
- Canny: effect of \(\sigma, T_H, T_L\); NMS logic.
- Harris: role of k and window size; eigenvalue interpretation.
- Fourier vs Euler (complex exponentials): \(e^{j\theta}=\cos\theta + j\sin\theta\).
- Color & shading: Lambertian vs specular; color spaces.

---

## Algorithms (Step-by-Step)

### Canny Edge Detection

1. Convert to grayscale; normalize.
2. Convolve with Gaussian of \(\sigma\) (choose by noise level/FWHM).
3. Compute \(G_x,G_y\) (Sobel/Scharr); magnitude and orientation.
4. Non-maximum suppression along quantized directions {0°,45°,90°,135°}.
5. Hysteresis thresholding with \(T_H,T_L=0.4\,T_H\) (rule of thumb).
6. Output connected edge map.

### Harris Corner Detection

1. Smooth image with Gaussian (for stability).
2. Compute gradients \(I_x,I_y\).
3. For each pixel, form M by Gaussian-weighted sums of \(I_x^2, I_y^2, I_x I_y\).
4. Compute \(R=\det(M)-k\,\operatorname{trace}^2\).
5. Threshold R; apply NMS; keep top N corners.

### Planar Camera Calibration (Zhang’s)

1. Capture ≥3 images of a planar pattern at different orientations.
2. For each image, estimate homography H from point correspondences.
3. Build linear system in \(B=K^{-\top}K^{-1}\) using orthogonality and equal-norm constraints.
4. Solve for B (SVD), then recover K via Cholesky.
5. For each image, compute \(r_1,r_2,t\) and normalize.
6. Optionally refine by non-linear bundle adjustment minimizing reprojection error.

---

## Differences and Comparisons

- Perspective vs Orthographic: perspective has depth-dependent scale (1/Z); orthographic parallel rays, constant scale.
- Lambertian vs Specular: Lambertian depends on \(n\cdot l\) only; specular depends on viewer direction (\(r\cdot v\)) and shininess m.
- Fourier vs Spatial: convolution ↔ multiplication; edges are high frequency; smoothing attenuates high frequency.
- Sobel vs Scharr: Scharr provides better rotational symmetry of gradient.

---

## Cheat Sheet

### Key Formulas

- Projection: \(s\,\mathbf{p}=K[R|t]\,\mathbf{P}\)
- Homography (plane): \(\mathbf{p}' \sim H\,\mathbf{p}\)
- DFT 2D: \(F[k,l]=\sum_m\sum_n f[m,n] e^{-j2\pi(km/M+ln/N)}\)
- Gradient magnitude: \(\sqrt{G_x^2+G_y^2}\)
- Laplacian: \(\nabla^2 f = f_{xx}+f_{yy}\)
- Harris: \(R=\det(M)-k\,\operatorname{trace}^2\)
- Optical flow: \(I_x u + I_y v + I_t = 0\)
- Stereo depth: \(Z=fB/d\)

### Quick Parameters

- Canny: \(\sigma\approx1{-}2\) (in px), \(T_L=0.4T_H\).
- Harris: \(k\in[0.04,0.06]\), window 5×5 to 7×7, Gaussian weight.

### Mini-Checklist

- Normalize intensities before filtering.
- Use separable Gaussian for speed: 1D conv twice.
- For calibration, spread orientations; fix lens distortion in refinement.

---

Prepared for fast revision; adapt numbers to your images if you run calculations.


