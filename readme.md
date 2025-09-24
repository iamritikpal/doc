## Computer Vision Notes (Units 1–3)

Designed for quick exam prep with plain-language explanations, ASCII diagrams, key formulas, derivations, and solved numericals. Cross-checks the highlighted exam topics: camera calibration, linear transforms, Canny, Harris, Euler/Fourier, color & shading.

---

### Unit 1 — Image Formation, Geometry, and Cameras

#### 1) Pinhole Camera: from 3D to 2D

Idea: Light rays pass through a single point (center of projection) and hit the image plane.

```
World point P(X,Y,Z)
           ^   Z-axis (optical axis)
           |         image plane at Z = +f
           |                (u,v)
     C •---+-----------------+------------------>
           |                 |
           |                 |
          (principal axis)   |
```

Calibrated perspective equations (image in metric units):

\[ x = f\,\frac{X}{Z},\quad y = f\,\frac{Y}{Z} \]

Homogeneous form with intrinsics/extrinsics:

\[ s\,\mathbf{p} = \mathbf{K}\,[\,\mathbf{R}\,|\,\mathbf{t}\,] \mathbf{P},\quad \mathbf{p}=(u,v,1)^\top,\; \mathbf{P}=(X,Y,Z,1)^\top \]

Where intrinsics

\[ \mathbf{K}=\begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0&0&1 \end{bmatrix} \]

- \(f_x = f / p_x\), \(f_y = f / p_y\) convert from meters to pixels.
- \(c_x,c_y\) principal point; \(s\) skew (≈0 for modern sensors).

Lens distortion (often corrected): radial \(k_1,k_2,k_3\) and tangential \(p_1,p_2\).

ASCII intuition: perspective vs orthographic

```
Perspective:        Orthographic:
   \\|//                ||
    \|/                ||      parallel rays
     •  <- center      ||
     |                 ||
  size ~ 1/Z           size ≈ constant (no foreshortening)
```

Orthographic and weak perspective:

- Orthographic: \(u = s_x X + c_x\), \(v = s_y Y + c_y\).
- Weak perspective: \(u=\alpha X + c_x\), \(v=\alpha Y + c_y\), with \(\alpha=f/\bar Z\).

#### 2) Linear Transformations (2D in the image plane)

Using homogeneous coordinates \((x,y,1)^\top\):

- Rotation: \(R_\theta = \begin{bmatrix}\cos\theta&-\sin\theta&0\\\sin\theta&\cos\theta&0\\0&0&1\end{bmatrix}\)
- Scaling: \(S = \operatorname{diag}(s_x,s_y,1)\)
- Shear: \(H_x=\begin{bmatrix}1&k&0\\0&1&0\\0&0&1\end{bmatrix}\)
- Translation: \(T=\begin{bmatrix}1&0&t_x\\0&1&t_y\\0&0&1\end{bmatrix}\)
- Affine = any product of the above. Projective adds last-row terms \([h_{31},h_{32},1]\).

Composition rule: last applied matrix multiplies on the left in homogeneous form.

#### 3) Euler Angles for Camera Orientation (ZYX convention)

Rotate world into camera via \(R=R_z(\gamma) R_y(\beta) R_x(\alpha)\):

\[
R_x=\begin{bmatrix}1&0&0\\0&\cos\alpha&-\sin\alpha\\0&\sin\alpha&\cos\alpha\end{bmatrix},\quad
R_y=\begin{bmatrix}\cos\beta&0&\sin\beta\\0&1&0\\-\sin\beta&0&\cos\beta\end{bmatrix},\quad
R_z=\begin{bmatrix}\cos\gamma&-\sin\gamma&0\\\sin\gamma&\cos\gamma&0\\0&0&1\end{bmatrix}
\]

Tip: avoid gimbal lock in optimization by using axis–angle or quaternions, but Euler is common in derivations.

#### 4) Radiometry, Light, and Shading

- Radiance \(L\) is conserved along a ray. Sensor irradiance \(E\propto L\) → pixel intensity.
- Lambertian: \(I = k_d\,\max(0, n\cdot l)\).
- Phong specular: \(I_s = k_s\,\max(0, r\cdot v)^m\).

Color spaces (must-know conversions):

- RGB→Gray: \(Y=0.299R+0.587G+0.114B\)
- RGB→HSV (concept): H from angle in (R,G,B) chroma plane, S = chroma/value, V = max(R,G,B).
- YCbCr: separates luma Y and chroma (Cb,Cr) for compression.

#### 5) Camera Calibration (Planar Zhang’s Method)

Given multiple views of a checkerboard (Z=0 in world plane): estimate homographies \(H_i\) and solve for intrinsics \(K\).

Key relation: columns of \(H_i\) are \(h_1,h_2,h_3\). Then

\[ \lambda_i r_1 = K^{-1} h_1,\quad \lambda_i r_2 = K^{-1} h_2,\quad \lambda_i t = K^{-1} h_3 \]

Orthogonality constraints yield linear equations in \(B=K^{-\top}K^{-1}\):

\[
v_{12}^\top b = 0,\quad (v_{11}-v_{22})^\top b = 0
\]

where each \(v_{pq}\) is built from \(h_p,h_q\). Stack ≥3 images → solve for b by SVD → recover \(K\) (Cholesky). Then compute \(R_i,t_i\), and refine all parameters by non-linear minimization of reprojection error.

---

### Unit 2 — Image Processing and Features

#### 1) Sampling, Convolution, and Fourier

2D DFT:

\[ F[k,l] = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} f[m,n] e^{-j 2\pi (km/M + ln/N)} \]

Inverse DFT restores the signal. Useful identities:

- Convolution theorem: \(\mathcal{F}\{f*h\} = F\cdot H\)
- Derivative in frequency: \(\mathcal{F}\{\partial f/\partial x\} = (j2\pi u)F(u,v)\)

Complex exponentials (Euler): \(e^{j\theta}=\cos\theta+j\sin\theta\).

#### 2) Smoothing and Differentiation

- Gaussian kernel \(G_\sigma(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-x^2/2\sigma^2}\) (separable in 2D).
- Laplacian: \(\nabla^2 f = f_{xx}+f_{yy}\); LoG detects zero-crossings of second derivative.

#### 3) Canny Edge Detector (with math)

Goal: thin, well-localized, low-noise edges.

Steps:

1. Smooth: \(I_s=I*G_\sigma\).
2. Gradient: \(G_x,G_y\) via Sobel/Scharr. Magnitude \(M=\sqrt{G_x^2+G_y^2}\); orientation \(\theta=\operatorname{atan2}(G_y,G_x)\).
3. Non-maximum suppression: keep a pixel only if its M is larger than its two neighbors along \(\theta\).
4. Hysteresis: thresholds \(T_H,T_L\) with \(T_L\approx0.4T_H\). Track weak edges connected to strong ones.

ASCII pipeline

```
Image → Gaussian(σ) → Gradients → NMS(thin) → Hysteresis → Final edges
```

#### 4) Harris Corner Detector (derivation intuition)

Auto-correlation of patch under small shift \(\Delta=(u,v)\):

\[ E(\Delta) \approx [u\ v] \; M \; [u\ v]^\top \]

with

\[ M = \sum_W \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} \]

Let eigenvalues of M be \(\lambda_1,\lambda_2\). Cases:

- flat: both small → no feature
- edge: one large, one small → edge
- corner: both large → corner

Practical score: \(R=\det(M)-k\,\operatorname{trace}(M)^2 = \lambda_1\lambda_2 - k(\lambda_1+\lambda_2)^2\) with \(k\in[0.04,0.06]\).

Post-processing: Gaussian weighting in M, threshold R, Non-Maximum Suppression.

#### 5) Texture Features (quick)

- GLCM statistics; LBP; Gabor filters at multiple scales/orientations.

---

### Unit 3 — Motion, Stereo, and Shape

#### 1) Optical Flow

Brightness constancy + small motion → Optical Flow Constraint Equation (OFCE):

\[ I_x u + I_y v + I_t = 0 \]

Lucas–Kanade in a window W solves

\[
\begin{bmatrix}\sum I_x^2 & \sum I_x I_y\\ \sum I_x I_y & \sum I_y^2\end{bmatrix}
\begin{bmatrix}u\\v\end{bmatrix}
= -\begin{bmatrix}\sum I_x I_t\\ \sum I_y I_t\end{bmatrix}
\]

Good features to track ≈ Harris corners (matrix well-conditioned).

#### 2) Stereo Depth (rectified)

Disparity \(d = u_L - u_R\). Depth

\[ Z = \frac{f B}{d} \]

where \(B\) baseline. Larger disparity → closer object.

#### 3) Shape-from-Shading (Lambertian)

\(I(x,y) = k_d\,\max(0, n(x,y)\cdot l)\) constrains surface normals; requires regularization for unique shapes.

---

## Solved Numerical Examples

### A) Camera Calibration (single planar view; intrinsics)

Image size 1280×720, principal point \((c_x,c_y)=(640,360)\), zero skew. Checker cell size 20 mm. Four corners (mm → px):

Plane: (0,0), (60,0), (60,40), (0,40)

Image: (512,290), (900,300), (880,520), (510,500)

1) Compute homography H (DLT). Suppose

\[ H=\begin{bmatrix}0.85&0.45&512\\ 0.02&1.10&290\\ 8e{-4}&9e{-4}&1\end{bmatrix} \]

2) With \(K=\begin{bmatrix}f_x&0&640\\0&f_y&360\\0&0&1\end{bmatrix}\), enforce

\(r_1 = \lambda K^{-1} h_1\), \(r_2 = \lambda K^{-1} h_2\), with \(r_1^\top r_2=0\) and \(\lVert r_1\rVert=\lVert r_2\rVert\). Solving gives \(f_x\approx1180\) px, \(f_y\approx1170\) px. Then \(t= K^{-1} h_3 / \lambda\).

Answer: \(K=\begin{bmatrix}1180&0&640\\0&1170&360\\0&0&1\end{bmatrix}\).

Tip: Use ≥10 images and refine with non-linear least squares for accuracy; include distortion parameters.

### B) Fourier Example (2D separable box blur)

Box kernel \(h[m,n]=\frac{1}{9}\) for \(m,n\in\{-1,0,1\}\). Its DFT magnitude is

\[ |H(u,v)| = \left|\frac{\sin(3\pi u)}{3\sin(\pi u)}\right| \cdot \left|\frac{\sin(3\pi v)}{3\sin(\pi v)}\right| \]

Observation: strong attenuation of high frequencies → edge smoothing.

### C) Harris on a small patch (numeric)

Suppose sums over a 5×5 window give \(\sum I_x^2=520\), \(\sum I_y^2=480\), \(\sum I_x I_y=30\).

\[ M=\begin{bmatrix}520&30\\30&480\end{bmatrix},\quad \det(M)=520\cdot480-30^2=249,\!\,100,\quad \operatorname{tr}=1000 \]

With \(k=0.05\): \(R=249100 - 0.05\times 10^6 = 249100 - 50000 = 199100 > 0\) → strong corner.

---

## Step-by-Step Algorithms (copy-ready)

### Canny

1) Grayscale and normalize to [0,1].
2) Smooth with Gaussian (σ≈1–2 px; larger for noisy images).
3) Compute gradients with Sobel/Scharr → M, θ.
4) Non-maximum suppression along θ (quantize to 0°,45°,90°,135°).
5) Hysteresis with TH and TL=0.4·TH; BFS/DFS track connectivity.
6) Optional: thin with morphological pruning.

### Harris

1) Gaussian blur (σ≈1). 2) Gradients Ix, Iy. 3) For each pixel, form M using Gaussian weights. 4) R=det(M)−k·trace², k∈[0.04,0.06]. 5) Threshold and NMS. 6) Keep top N corners.

### Planar Calibration (Zhang)

1) Capture ≥3 tilted views of a checkerboard; detect corners subpixel.
2) For each image, compute homography H via DLT from ≥4 correspondences.
3) Build Vb=0 to solve B=K^{−T}K^{−1}. 4) Recover K (Cholesky).
5) For each image: r1=λK^{−1}h1, r2=λK^{−1}h2, r3=r1×r2; t=λK^{−1}h3. Normalize ri.
6) Jointly refine all parameters (bundle adjustment) including distortion.

---

## Comparisons to Memorize

- Perspective vs Orthographic: perspective scale ∝1/Z, lines not parallel preserved only in orthographic.
- Lambertian vs Specular: Lambertian independent of viewer; specular depends on view dir and shininess.
- Sobel vs Scharr: Scharr has better rotational isotropy (preferred for Canny gradients).
- Fourier vs Spatial domain: convolution ↔ multiplication; smoothing reduces high-frequency magnitude.

---

## Cheat Sheet (1-page)

- Projection: \(s\,p=K[R|t]P\). Plane homography \(p'\sim H p\).
- Intrinsics: \(K=\begin{bmatrix}f_x&s&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}\).
- DFT2: \(F[k,l]=\sum f[m,n]e^{-j2\pi(km/M+ln/N)}\); inverse analogous.
- Gradients: \(M=\sqrt{G_x^2+G_y^2}\), \(\theta=\operatorname{atan2}(G_y,G_x)\).
- Laplacian: \(\nabla^2 f=f_{xx}+f_{yy}\). LoG=\(\nabla^2(G_\sigma*f)\).
- Harris: \(M=\sum[\![I_x^2,I_xI_y;I_xI_y,I_y^2]\!]\), \(R=\det(M)-k\,\operatorname{tr}(M)^2\).
- OFCE: \(I_x u+I_y v+I_t=0\). Lucas–Kanade normal equations above.
- Stereo depth: \(Z=fB/d\).
- RGB→Gray: \(0.299R+0.587G+0.114B\).

Parameter tips: Canny TL≈0.4·TH; Harris k≈0.04–0.06; Gaussian σ≈1–2 px.

---

End of notes. Use alongside your lecture slides for examples and diagrams.


