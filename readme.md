# Computer Vision Complete Study Guide (Units 1-3)
## 📚 Comprehensive Notes for Mid-Semester Exam

This guide covers all topics from your syllabus with detailed explanations, visual diagrams, mathematical derivations, and solved examples. Perfect for exam preparation!

---

## 📖 Table of Contents
1. [Unit 1: Image Formation Models & Camera Systems](#unit-1-image-formation-models--camera-systems)
2. [Unit 2: Image Processing & Feature Extraction](#unit-2-image-processing--feature-extraction) 
3. [Unit 3: Motion Estimation & 3D Vision](#unit-3-motion-estimation--3d-vision)
4. [Solved Numerical Examples](#solved-numerical-examples)
5. [Step-by-Step Algorithms](#step-by-step-algorithms)
6. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

# Unit 1: Image Formation Models & Camera Systems

## 1.1 Imaging Systems Overview

### Monocular vs Binocular Imaging

```
MONOCULAR SYSTEM          BINOCULAR SYSTEM
     (Single Camera)           (Stereo Cameras)

        Object                    Object
          |                        |
          |                   _____|_____
          v                  /           \
      [Camera]              /             \
          |               [Left]         [Right]
          |                 |             |
      [Image]           [Left Image] [Right Image]
                             |             |
                             \             /
                              \___________/
                                   |
                               [Depth Map]
```

**Key Differences:**
- **Monocular**: Single viewpoint, loses depth information
- **Binocular**: Two viewpoints, enables depth perception through disparity

## 1.2 Camera Projection Models

### Perspective Projection (Pinhole Camera)

The fundamental model for most cameras:

```
3D WORLD COORDINATE SYSTEM → 2D IMAGE PLANE

                    Z (optical axis)
                    ↑
                    |    
                    |    P(X,Y,Z)
                    |   /
                    |  /
                    | /
    Y ←-------------|----------→ 
                    |/           
                    C (camera center)
                    |
                    |
                    |___________→ X
                    
                    |
                    | f (focal length)
                    |
                ----+----  ← Image plane
                    |
                   p(x,y) ← projected point
```

**Mathematical Formulation:**

**Basic Perspective Equations:**
```
x = f × (X/Z)
y = f × (Y/Z)
```

**Homogeneous Coordinates (Complete Model):**
```
s × [u]   [fx  s  cx] [r11 r12 r13 tx] [X]
    [v] = [ 0 fy  cy] [r21 r22 r23 ty] [Y]
    [1]   [ 0  0   1] [r31 r32 r33 tz] [Z]
                                        [1]
```

Where:
- **Intrinsic Matrix K**: Camera internal parameters
- **Extrinsic Matrix [R|t]**: Camera pose in world coordinates

### Orthographic Projection

Simplified model for distant objects:

```
PERSPECTIVE vs ORTHOGRAPHIC

Perspective:           Orthographic:
    \  |  /               |  |  |
     \ | /                |  |  |
      \|/                 |  |  |
       •  ← pinhole       |  |  |
       |                  |  |  |
   [Image]            [Image]

Size ∝ 1/Z            Size = constant
```

**Mathematical Model:**
```
x = sₓ × X + cx
y = sᵧ × Y + cy
```

**When to Use:**
- Telephoto lenses (small field of view)
- Objects very far from camera
- Medical imaging, satellite imagery

### Weak Perspective

Compromise between perspective and orthographic:

```
x = α × X + cx
y = α × Y + cy

where α = f/Z̄ (average depth)
```

## 1.3 Camera Intrinsic Parameters

### The Intrinsic Matrix K

```
K = [fx  s  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```

**Parameter Meanings:**

1. **fx, fy**: Focal lengths in pixel units
   - fx = f/px (f in mm, px = pixel size in mm)
   - fy = f/py
   - Usually fx ≈ fy for square pixels

2. **cx, cy**: Principal point (optical center)
   - Ideally at image center: (width/2, height/2)
   - Can be offset due to manufacturing

3. **s**: Skew parameter
   - Usually ≈ 0 for modern cameras
   - Non-zero if pixel grid not rectangular

### Lens Distortion

Real lenses introduce distortion:

```
BARREL DISTORTION        PINCUSHION DISTORTION
    (negative k₁)            (positive k₁)

     ___                         ___
    (   )                       |   |
   (     )          vs         (     )
    (   )                       |   |
     ___                         ---
```

**Distortion Model:**
```
Radial: δr = k₁r³ + k₂r⁵ + k₃r⁷
Tangential: δx = 2p₁xy + p₂(r² + 2x²)
           δy = p₁(r² + 2y²) + 2p₂xy
```

## 1.4 Camera Extrinsic Parameters

### Rotation and Translation

**Extrinsic parameters define camera pose:**
- **R**: 3×3 rotation matrix (3 DOF)
- **t**: 3×1 translation vector (3 DOF)

```
WORLD TO CAMERA TRANSFORMATION

World Point Pw → Camera Point Pc
Pc = R × Pw + t

Or in homogeneous form:
[Pc] = [R  t] [Pw]
[ 1]   [0  1] [ 1]
```

### Euler Angles (ZYX Convention)

**Sequential Rotations:**
1. Rotate by γ around Z-axis
2. Rotate by β around Y-axis  
3. Rotate by α around X-axis

```
Rx(α) = [1    0      0   ]    Ry(β) = [cos β   0  sin β]
        [0  cos α -sin α]              [  0     1    0  ]
        [0  sin α  cos α]              [-sin β  0  cos β]

Rz(γ) = [cos γ -sin γ  0]
        [sin γ  cos γ  0]
        [  0     0     1]

Final: R = Rz(γ) × Ry(β) × Rx(α)
```

**Gimbal Lock Warning:** Avoid β = ±90° in optimization!

## 1.5 Linear Transformations in 2D

### Transformation Types

```
TRANSFORMATION HIERARCHY

Projective (8 DOF)
    |
Affine (6 DOF)
    |
Similarity (4 DOF)  
    |
Euclidean (3 DOF)
    |
Translation (2 DOF)
```

**1. Translation:**
```
T = [1  0  tx]
    [0  1  ty]
    [0  0   1]
```

**2. Rotation:**
```
R = [cos θ  -sin θ  0]
    [sin θ   cos θ  0]
    [ 0       0     1]
```

**3. Scaling:**
```
S = [sx  0   0]
    [ 0  sy  0]
    [ 0   0  1]
```

**4. Shearing:**
```
Hx = [1  shx  0]     Hy = [ 1   0   0]
     [0   1   0]          [shy  1   0]
     [0   0   1]          [ 0   0   1]
```

**5. Affine (combines all above):**
```
A = [a11  a12  tx]
    [a21  a22  ty]
    [ 0    0   1]
```

**6. Projective (Homography):**
```
H = [h11  h12  h13]
    [h21  h22  h23]
    [h31  h32  h33]
```

### Homography Between Planes

**Key Application:** Relating two views of the same plane

```
PLANAR HOMOGRAPHY EXAMPLE

View 1:                View 2:
+-------+              +-------+
|   A   |              |  A'   |
| D   B | ----H----->  |D'   B'|
|   C   |              |  C'   |
+-------+              +-------+

A' = H × A (in homogeneous coordinates)
```

**Mathematical Derivation:**
For a plane π: nᵀX + d = 0, the homography between two camera views is:

```
H = K' × (R - t×nᵀ/d) × K⁻¹
```

## 1.6 Radiometry and Light

### Basic Radiometric Quantities

```
LIGHT MEASUREMENT CHAIN

Light Source → Surface → Camera
    |           |         |
Radiant      Reflected   Measured
Intensity    Radiance    Irradiance
```

**Key Quantities:**
1. **Radiance L** [W·sr⁻¹·m⁻²]: Power per unit area per solid angle
2. **Irradiance E** [W·m⁻²]: Power per unit area
3. **Radiant Intensity I** [W·sr⁻¹]: Power per solid angle

**Important Property:** Radiance is conserved along rays (no absorption)

### Surface Reflection Models

#### Lambertian (Diffuse) Reflection

Perfect diffuse reflector - appears equally bright from all viewing angles.

```
LAMBERTIAN SURFACE

    Light       Normal
      ↘         ↑
       ↘        |
        ↘       |
    θ    ↘      |
          ↘_____|
           Surface

Intensity I = ρ × I₀ × max(0, cos θ)
            = ρ × I₀ × max(0, n⃗ · l⃗)
```

Where:
- ρ: albedo (reflectance coefficient)
- I₀: incident light intensity
- θ: angle between surface normal and light direction

#### Specular Reflection (Phong Model)

Shiny surfaces with mirror-like reflection component.

```
SPECULAR REFLECTION

    Light    Normal    View
      ↘       ↑        ↗
       ↘      |       ↗
        ↘     |      ↗
         ↘    |     ↗
          ↘___|____↗
           Surface
              ↑
           Reflection

I = ks × I₀ × max(0, r⃗ · v⃗)ᵐ
```

Where:
- ks: specular coefficient
- r⃗: perfect reflection direction
- v⃗: viewing direction  
- m: shininess exponent

**Combined Model:**
```
I_total = I_ambient + I_diffuse + I_specular
        = ka×Ia + kd×I₀×(n⃗·l⃗) + ks×I₀×(r⃗·v⃗)ᵐ
```

## 1.7 Color Spaces and Models

### RGB Color Space

**Device-dependent primary colors:**

```
RGB COLOR CUBE

        White(1,1,1)
           +--------+
          /|       /|
    Cyan +--------+ | Magenta
        /  |      / |
       +--------+  |
    Blue|   +-----|--+ Yellow
        |  /      | /
        | /       |/
        +--------+
      Black     Red
     (0,0,0)
```

**Properties:**
- Additive color model
- Device dependent (different monitors ≠ same color)
- Not perceptually uniform

### HSV Color Space

**More intuitive for humans:**

```
HSV CYLINDER

         White
           ↑
           |  ← Value (brightness)
           |
     ------+------  ← Saturation (purity)
    /             \
   /               \
  /                 \
 /        Hue        \
/     (color angle)   \
+---------------------+
         Black
```

**Conversion from RGB:**
```
V = max(R, G, B)
S = (V - min(R,G,B)) / V  (if V ≠ 0)
H = angle in color wheel based on which component is max
```

### YCbCr Color Space

**Separates luminance from chrominance:**

```
Y  = 0.299×R + 0.587×G + 0.114×B  (luminance)
Cb = 0.564×(B - Y)                 (blue chroma)
Cr = 0.713×(R - Y)                 (red chroma)
```

**Advantages:**
- Human vision more sensitive to Y than Cb,Cr
- Enables chroma subsampling for compression
- Used in JPEG, MPEG

## 1.8 Camera Calibration (Zhang's Method)

### Problem Statement

**Goal:** Estimate intrinsic matrix K and extrinsic parameters R,t from images of a known calibration pattern.

```
CALIBRATION SETUP

Known 3D Pattern    →    Camera    →    Observed 2D Points
(checkerboard)                          
                    
World Coordinates   →   Projection  →   Image Coordinates
    (X,Y,Z)                               (u,v)
```

### Zhang's Planar Method

**Advantages:**
- Only requires 2D calibration pattern
- Multiple orientations of same pattern
- Robust and widely used

**Step-by-Step Process:**

#### Step 1: Capture Multiple Images
```
IMAGE 1:        IMAGE 2:        IMAGE 3:
+-----+         +-----+         +-----+
| ╱╲  |         |╱   ╲|         |    ╱|
|╱  ╲ |   vs    |    ╲|   vs    |   ╱ |
|   ╲╱|         |     ╲|         |  ╱  |
+-----+         +-----+         +-----+
(frontal)       (tilted)        (rotated)
```

#### Step 2: Detect Corners and Estimate Homographies

For each image i, find homography Hi that maps world plane points to image points:

```
[ui]     [Xi]
[vi] = Hi[Yi]  (in homogeneous coordinates)
[1 ]     [1 ]
```

#### Step 3: Solve for Intrinsic Parameters

**Key Insight:** Each homography provides constraints on K.

If Hi = [h1 h2 h3], then:
```
Hi = λ × K × [r1 r2 t]
```

**Orthogonality Constraints:**
```
r1ᵀ × r2 = 0     (orthogonal)
||r1|| = ||r2||  (unit vectors)
```

These lead to linear equations in matrix B = K⁻ᵀK⁻¹:

```
[h1ᵀ B h2] = 0
[(h1-h2)ᵀ B (h1-h2)] = 0
```

#### Step 4: Recover K from B

Once B is found, recover K using Cholesky decomposition:
```
B = K⁻ᵀK⁻¹
K⁻ᵀ = chol(B)
K = (K⁻ᵀ)⁻ᵀ
```

#### Step 5: Compute Extrinsics for Each Image

```
λ = 1/||K⁻¹h1||
r1 = λ × K⁻¹h1
r2 = λ × K⁻¹h2
r3 = r1 × r2
t  = λ × K⁻¹h3
```

#### Step 6: Refine with Bundle Adjustment

Minimize total reprojection error:
```
min Σᵢⱼ ||pᵢⱼ - π(K, Rᵢ, tᵢ, Pⱼ)||²
```

---

# Unit 2: Image Processing & Feature Extraction

## 2.1 Digital Image Fundamentals

### Image Representation

```
CONTINUOUS vs DISCRETE

Continuous Image f(x,y)    →    Discrete Image f[m,n]
                                
     y ↑                           n ↑
       |                            |
       |                            |
    ---+--→ x                    ---+--→ m
                                    
Infinite resolution              Finite grid (M×N pixels)
```

**Sampling Process:**
```
f[m,n] = f(m×Δx, n×Δy)
```

Where Δx, Δy are sampling intervals.

### Nyquist Sampling Theorem

**Critical Rule:** Sample rate ≥ 2 × highest frequency to avoid aliasing

```
ALIASING EXAMPLE

High-freq signal:  ∿∿∿∿∿∿∿∿
Undersampled:      · · · ·
Reconstructed:     ∼   ∼   ∼   (false low frequency!)

Proper sampling:   ∿∿∿∿∿∿∿∿
Sufficient rate:   ·····
Reconstructed:     ∿∿∿∿∿∿∿∿  (correct!)
```

## 2.2 Fourier Transform in 2D

### Mathematical Definition

**2D Continuous Fourier Transform:**
```
F(u,v) = ∫∫ f(x,y) × e^(-j2π(ux+vy)) dx dy
```

**2D Discrete Fourier Transform (DFT):**
```
F[k,l] = (1/MN) × Σₘ₌₀^(M-1) Σₙ₌₀^(N-1) f[m,n] × e^(-j2π(km/M + ln/N))
```

### Frequency Domain Interpretation

```
SPATIAL vs FREQUENCY DOMAIN

Spatial Domain:              Frequency Domain:
                            
+-------+                   +-------+
|  ∩∩   |                   |   ·   |  ← DC component
| ∩∩∩∩  |  ←→ DFT ←→        | ·   · |  ← Low frequencies  
|∩∩∩∩∩∩ |                   |  · ·  |  ← High frequencies
+-------+                   +-------+
(image)                     (spectrum)
```

**Key Properties:**

1. **DC Component:** F[0,0] = average brightness
2. **Low Frequencies:** Smooth variations, overall shape
3. **High Frequencies:** Edges, noise, fine details

### Important Fourier Properties

#### 1. Convolution Theorem
```
f ⊗ h  ←→  F × H
(spatial convolution = frequency multiplication)
```

#### 2. Differentiation Property
```
∂f/∂x  ←→  j2πu × F(u,v)
∂f/∂y  ←→  j2πv × F(u,v)
```

#### 3. Scaling Property
```
f(ax, by)  ←→  (1/|ab|) × F(u/a, v/b)
```

## 2.3 Linear Filtering

### Convolution Operation

**2D Convolution:**
```
g[m,n] = Σₖ Σₗ f[k,l] × h[m-k, n-l]
       = f ⊗ h
```

**Visual Interpretation:**
```
CONVOLUTION PROCESS

Original Image    Kernel         Result
+---+---+---+    +---+---+      +---+---+---+
| a | b | c |    |-1 | 0 | 1|   |   |   |   |
+---+---+---+  ⊗ +---+---+   = +---+---+---+
| d | e | f |    |-2 | 0 | 2|   |   | R |   |
+---+---+---+    +---+---+      +---+---+---+
| g | h | i |    |-1 | 0 | 1|   |   |   |   |
+---+---+---+    +---+---+      +---+---+---+

R = a×(-1) + b×0 + c×1 + d×(-2) + e×0 + f×2 + g×(-1) + h×0 + i×1
  = -a + c - 2d + 2f - g + i
```

### Common Linear Filters

#### 1. Gaussian Filter (Smoothing)

**1D Gaussian:**
```
G(x) = (1/√(2πσ²)) × e^(-x²/2σ²)
```

**2D Gaussian (Separable):**
```
G(x,y) = G(x) × G(y)

Discrete approximation (σ=1):
    1  [ 1  2  1 ]
   --- [ 2  4  2 ]
   16  [ 1  2  1 ]
```

**Properties:**
- Removes noise while preserving edges
- Separable (can apply 1D filters sequentially)
- Scale parameter σ controls blur amount

#### 2. Box Filter (Simple Averaging)

```
Box Filter (3×3):
    1  [ 1  1  1 ]
   --- [ 1  1  1 ]
    9  [ 1  1  1 ]
```

**Frequency Response:**
```
|H(u,v)| = |sin(3πu)/(3sin(πu))| × |sin(3πv)/(3sin(πv))|
```

#### 3. Derivative Filters

**Sobel X-derivative:**
```
Sx = [-1  0  1]     Sy = [-1 -2 -1]
     [-2  0  2]          [ 0  0  0]
     [-1  0  1]          [ 1  2  1]
```

**Scharr (Better rotational symmetry):**
```
Sx = [-3  0  3]     Sy = [-3 -10 -3]
     [-10 0 10]          [ 0   0  0]
     [-3  0  3]          [ 3  10  3]
```

#### 4. Laplacian (Second Derivative)

```
∇²f = ∂²f/∂x² + ∂²f/∂y²

Discrete Laplacian:
    [ 0 -1  0]      or    [-1 -1 -1]
    [-1  4 -1]            [-1  8 -1]
    [ 0 -1  0]            [-1 -1 -1]
```

**Laplacian of Gaussian (LoG):**
```
LoG = ∇²(G_σ ⊗ f) = (∇²G_σ) ⊗ f
```

## 2.4 Edge Detection

### What Are Edges?

**Definition:** Significant local changes in image intensity

```
EDGE TYPES

Step Edge:     Ramp Edge:     Ridge Edge:
    |              /              /\
    |             /              /  \
----+         ---/           ---/    \---
    |                              
    |                              

Intensity profiles showing different edge characteristics
```

### Canny Edge Detection Algorithm

**Goals:**
1. **Good Detection:** Find all real edges, minimize false positives
2. **Good Localization:** Edges close to true edge locations  
3. **Single Response:** One response per edge

#### Step 1: Noise Reduction

Apply Gaussian smoothing:
```
I_smooth = G_σ ⊗ I
```

**Effect of σ:**
- Small σ: Preserves details, more noise
- Large σ: Removes noise, loses fine edges

```
σ = 0.5:  ∿∿∿∿∿∿  (noisy but detailed)
σ = 1.0:  ∼∼∼∼∼∼  (balanced)
σ = 2.0:  ∼  ∼  ∼  (smooth but blurred)
```

#### Step 2: Gradient Computation

```
Gx = Sx ⊗ I_smooth    (horizontal gradients)
Gy = Sy ⊗ I_smooth    (vertical gradients)

Magnitude: M = √(Gx² + Gy²)
Direction: θ = arctan2(Gy, Gx)
```

**Gradient Direction Quantization:**
```
      90°
       |
       |
135° ——+—— 45°
       |
       |
      0°

Quantize θ to: {0°, 45°, 90°, 135°}
```

#### Step 3: Non-Maximum Suppression

**Goal:** Thin thick edges to single pixel width

```
NMS PROCESS

Before NMS:        After NMS:
                  
  ∿∿∿               |
 ∿∿∿∿∿       →      |
  ∿∿∿               |
                  
(thick edge)      (thin edge)
```

**Algorithm:**
For each pixel (i,j):
1. Find gradient direction θ[i,j]
2. Check neighbors along θ direction
3. Keep pixel only if M[i,j] ≥ both neighbors

#### Step 4: Double Thresholding

**Two thresholds:**
- **T_high:** Strong edge threshold
- **T_low:** Weak edge threshold (typically T_low = 0.4 × T_high)

```
THRESHOLDING CLASSIFICATION

M > T_high:   Strong edge (keep)
T_low < M < T_high: Weak edge (maybe keep)
M < T_low:    Non-edge (discard)
```

#### Step 5: Edge Tracking by Hysteresis

**Goal:** Connect weak edges to strong edges

```
HYSTERESIS LINKING

Strong edges: ████
Weak edges:   ░░░░
Background:   ....

Before:                After:
████....░░░░           ████████████
....████░░░░    →      ....████████
....░░░░....           ....████....
```

**Algorithm:**
1. Start from strong edge pixels
2. Follow connected weak edge pixels
3. Mark connected weak edges as final edges
4. Discard unconnected weak edges

### Canny Parameter Effects

```
PARAMETER TUNING GUIDE

σ (Gaussian width):
- Too small: Noisy edges
- Too large: Missing fine edges
- Typical: 0.5-2.0

T_high (Strong threshold):
- Too low: Many false edges
- Too high: Missing real edges
- Typical: 0.1-0.3 × max(gradient)

T_low (Weak threshold):
- Usually: 0.4 × T_high
- Lower ratio: More edge linking
- Higher ratio: Less edge linking
```

## 2.5 Corner Detection

### What Are Corners?

**Definition:** Points where image gradient changes direction rapidly

```
FEATURE CLASSIFICATION

Flat Region:    Edge:         Corner:
   ....         ||||           /|
   ....         ||||          / |
   ....         ||||         /  |

No gradient    1D gradient   2D gradient
```

### Harris Corner Detector

#### Mathematical Foundation

**Auto-correlation Function:**
```
E(u,v) = Σ w(x,y) × [I(x+u, y+v) - I(x,y)]²
```

Where w(x,y) is a window function (usually Gaussian).

**Taylor Expansion:**
```
I(x+u, y+v) ≈ I(x,y) + u×Ix + v×Iy

E(u,v) ≈ [u v] × M × [u]
                      [v]
```

**Structure Matrix M:**
```
M = Σ w(x,y) × [Ix²   IxIy]
                [IxIy  Iy² ]
```

#### Eigenvalue Analysis

**Matrix M has eigenvalues λ₁, λ₂:**

```
EIGENVALUE INTERPRETATION

Case 1: λ₁ ≈ 0, λ₂ ≈ 0
   → Flat region (no features)

Case 2: λ₁ >> λ₂ ≈ 0  or  λ₂ >> λ₁ ≈ 0
   → Edge (one dominant direction)

Case 3: λ₁ >> 0, λ₂ >> 0
   → Corner (two strong directions)
```

#### Harris Response Function

**Avoid eigenvalue computation:**
```
R = det(M) - k × trace(M)²
  = λ₁λ₂ - k(λ₁ + λ₂)²

where k ∈ [0.04, 0.06]
```

**Response Interpretation:**
```
R > 0:  Corner
R < 0:  Edge  
R ≈ 0:  Flat region
```

#### Complete Harris Algorithm

```
HARRIS CORNER DETECTION PIPELINE

1. Smooth image with Gaussian (σ ≈ 1)
2. Compute gradients: Ix, Iy
3. For each pixel:
   - Compute structure matrix M (with Gaussian weighting)
   - Calculate R = det(M) - k×trace(M)²
4. Apply threshold: keep pixels where R > threshold
5. Non-maximum suppression
6. Return top N corners
```

### Corner Quality Assessment

**Good Corners Have:**
- High Harris response
- Well-distributed in image
- Stable across scales
- Repeatable across viewpoints

```
CORNER QUALITY RANKING

Excellent: ┼     Good: ╋      Poor: ╬
          ┼           ╋            ╬
```

## 2.6 Texture Analysis

### Gray Level Co-occurrence Matrix (GLCM)

**Concept:** Analyze spatial relationships between pixel intensities

```
GLCM CONSTRUCTION

Image:          Direction (1,0):    GLCM:
1 2 1           Pairs: (1,2), (2,1)    
2 1 2           (2,1), (1,2)        i\j  1  2
1 2 1           (1,2), (2,1)        1   [0  4]
                                    2   [4  0]
```

**Common GLCM Features:**
1. **Contrast:** Σᵢⱼ(i-j)² × P(i,j)
2. **Energy:** Σᵢⱼ P(i,j)²
3. **Homogeneity:** Σᵢⱼ P(i,j)/(1+|i-j|)
4. **Correlation:** Measures linear dependency

### Local Binary Patterns (LBP)

**Rotation-invariant texture descriptor:**

```
LBP COMPUTATION

Neighborhood:    Thresholded:    Binary:    LBP Value:
6  5  2         1  1  0         11000111    199
7  6  1    →    1  ·  0    →    
9  8  7         1  1  1         

Center = 6, compare neighbors:
5≥6? Yes(1), 2≥6? No(0), 1≥6? No(0), etc.
```

---

# Unit 3: Motion Estimation & 3D Vision

## 3.1 Optical Flow

### Brightness Constancy Assumption

**Fundamental assumption:** Pixel intensities remain constant as they move

```
OPTICAL FLOW CONCEPT

Frame t:           Frame t+1:         Motion Vectors:
  ●                  ●                   ●→
    ●                  ●       =           ●→
      ●                  ●                   ●→

I(x,y,t) = I(x+u, y+v, t+1)
```

### Optical Flow Constraint Equation (OFCE)

**Derivation using Taylor expansion:**
```
I(x+u, y+v, t+1) ≈ I(x,y,t) + u×∂I/∂x + v×∂I/∂y + ∂I/∂t

Setting equal (brightness constancy):
u×Ix + v×Iy + It = 0
```

**This is one equation with two unknowns (u,v)!**

### Lucas-Kanade Method

**Solution:** Assume flow is constant in local neighborhood

```
LUCAS-KANADE WINDOW

For each pixel in window W:
u×Ix + v×Iy + It = 0

Matrix form:
[Ix₁  Iy₁] [u]   [-It₁]
[Ix₂  Iy₂] [v] = [-It₂]
[... ...] [.]   [...]
[IxN  IyN]       [-ItN]

A × [u,v]ᵀ = b
```

**Least squares solution:**
```
[u] = (AᵀA)⁻¹ × Aᵀb
[v]

Where:
AᵀA = [ΣIx²   ΣIxIy]
      [ΣIxIy  ΣIy² ]
```

**Window Selection:**
```
Good Window:        Bad Window:
████░░░░            ........
████░░░░            ........
░░░░████            ........
░░░░████            ........
(corners/texture)   (uniform)
```

### Pyramidal Lucas-Kanade

**Problem:** Large motions violate small displacement assumption

**Solution:** Multi-scale approach

```
PYRAMID STRUCTURE

Level 2:  [▓▓]     ← Coarse level (large motion)
          [▓▓]
          
Level 1:  [▓▓▓▓]   ← Medium level  
          [▓▓▓▓]
          [▓▓▓▓]
          [▓▓▓▓]
          
Level 0:  [████████] ← Fine level (refinement)
          [████████]
          [████████]
          [████████]
          [████████]
          [████████]
          [████████]
          [████████]
```

**Algorithm:**
1. Build image pyramids for both frames
2. Start at coarsest level, estimate flow
3. Propagate and refine at finer levels
4. Final result combines all levels

## 3.2 Stereo Vision

### Stereo Geometry

```
STEREO CAMERA SETUP

        PL(XL,YL,ZL)  ← World point
           /    \
          /      \
         /        \
    Left Camera  Right Camera
        |           |
        |           |
    [Left Image] [Right Image]
       pL(uL,vL)   pR(uR,vR)
```

**Key Parameters:**
- **Baseline B:** Distance between camera centers
- **Focal length f:** Same for both cameras (rectified)
- **Disparity d:** Horizontal difference d = uL - uR

### Depth from Disparity

**Similar triangles relationship:**
```
Z/f = B/(uL - uR)

Therefore: Z = f×B/d
```

**Depth Resolution:**
```
dZ/dd = -f×B/d²

Implications:
- Close objects: Small d change → Large Z change
- Far objects: Small d change → Small Z change
```

### Stereo Rectification

**Goal:** Make epipolar lines horizontal

```
BEFORE RECTIFICATION       AFTER RECTIFICATION

Left:    Right:           Left:    Right:
  ╱╲      ╱╲                ——      ——
 ╱  ╲    ╱  ╲               ——      ——  
╱____╲  ╱____╲              ——      ——

Epipolar lines at angles   Horizontal epipolar lines
→ 2D search problem        → 1D search problem
```

### Correspondence Problem

**Challenge:** Match pixels between left and right images

**Constraints:**
1. **Epipolar:** Corresponding points lie on same horizontal line
2. **Ordering:** Relative order preserved (mostly)
3. **Uniqueness:** One-to-one correspondence
4. **Smoothness:** Nearby pixels have similar disparities

**Matching Costs:**
```
SAD: Σ|IL(x,y) - IR(x-d,y)|
SSD: Σ(IL(x,y) - IR(x-d,y))²
NCC: Correlation coefficient
```

## 3.3 Structure from Motion

### Fundamental Matrix

**Relates corresponding points in two views:**
```
p₂ᵀ × F × p₁ = 0
```

Where F encodes epipolar geometry.

### Essential Matrix

**For calibrated cameras:**
```
E = [t]× × R

Where [t]× is skew-symmetric matrix of translation
```

### 8-Point Algorithm

**Estimate F from ≥8 point correspondences:**
```
For each correspondence (p₁ᵢ, p₂ᵢ):
[u₁u₂  v₁u₂  u₂  u₁v₂  v₁v₂  v₂  u₁  v₁  1] × f = 0

Stack 8+ equations → solve Af = 0 (SVD)
```

---

# Solved Numerical Examples

## Example 1: Camera Calibration

**Problem:** Calibrate camera using checkerboard with 20mm squares.

**Given Data:**
- Image size: 1280×720 pixels
- Principal point estimate: (640, 360)
- 4 corner correspondences:

```
World Points (mm):    Image Points (pixels):
(0, 0)         →      (512, 290)
(60, 0)        →      (900, 300)  
(60, 40)       →      (880, 520)
(0, 40)        →      (510, 500)
```

**Solution Steps:**

### Step 1: Estimate Homography H

Set up DLT equations (8 equations from 4 points):
```
For each correspondence (X,Y) → (u,v):
-X  -Y  -1   0   0   0  uX  uY  u
 0   0   0  -X  -Y  -1  vX  vY  v
```

Solving gives (example result):
```
H = [0.85   0.45   512]
    [0.02   1.10   290]
    [8e-4  9e-4    1 ]
```

### Step 2: Extract Camera Parameters

From H = λK[r₁ r₂ t], with K having zero skew and known principal point:
```
K = [fx   0  640]
    [ 0  fy  360]
    [ 0   0    1]
```

**Orthogonality constraints:**
```
r₁ᵀr₂ = 0
||r₁|| = ||r₂|| = 1
```

This gives:
```
λ²(K⁻¹h₁)ᵀ(K⁻¹h₂) = 0
λ²||K⁻¹h₁||² = λ²||K⁻¹h₂||²
```

Solving these equations:
```
fx ≈ 1180 pixels
fy ≈ 1170 pixels
```

**Final Result:**
```
K = [1180   0  640]
    [   0 1170  360]
    [   0   0    1]
```

## Example 2: Harris Corner Detection

**Problem:** Compute Harris response for a 3×3 patch.

**Given gradients:**
```
Ix = [1  2  1]    Iy = [1  0 -1]
     [0  0  0]         [2  0 -2]
     [-1-2 -1]         [1  0 -1]
```

**Solution:**

### Step 1: Compute Structure Matrix Components
```
ΣIx² = 1² + 2² + 1² + 0² + 0² + 0² + (-1)² + (-2)² + (-1)²
     = 1 + 4 + 1 + 0 + 0 + 0 + 1 + 4 + 1 = 12

ΣIy² = 1² + 0² + (-1)² + 2² + 0² + (-2)² + 1² + 0² + (-1)²
     = 1 + 0 + 1 + 4 + 0 + 4 + 1 + 0 + 1 = 12

ΣIxIy = (1×1) + (2×0) + (1×-1) + (0×2) + (0×0) + (0×-2) + (-1×1) + (-2×0) + (-1×-1)
      = 1 + 0 - 1 + 0 + 0 + 0 - 1 + 0 + 1 = 0
```

### Step 2: Form Structure Matrix
```
M = [12   0]
    [ 0  12]
```

### Step 3: Compute Harris Response
```
det(M) = 12 × 12 - 0 × 0 = 144
trace(M) = 12 + 12 = 24

With k = 0.05:
R = det(M) - k × trace(M)²
  = 144 - 0.05 × 24²
  = 144 - 0.05 × 576
  = 144 - 28.8
  = 115.2
```

**Result:** R = 115.2 > 0 → **Strong Corner**

## Example 3: Optical Flow Computation

**Problem:** Compute optical flow using Lucas-Kanade method.

**Given 3×3 window gradients:**
```
Ix values: [2, 1, 3, 2, 4, 1, 3, 2, 1]
Iy values: [1, 3, 2, 4, 1, 3, 2, 1, 4]  
It values: [-2, -1, -3, -2, -4, -1, -3, -2, -1]
```

**Solution:**

### Step 1: Form Normal Equations
```
AᵀA = [ΣIx²   ΣIxIy]    Aᵀb = [-ΣIxIt]
      [ΣIxIy  ΣIy² ]          [-ΣIyIt]
```

### Step 2: Compute Sums
```
ΣIx² = 4+1+9+4+16+1+9+4+1 = 49
ΣIy² = 1+9+4+16+1+9+4+1+16 = 61  
ΣIxIy = 2+3+6+8+4+3+6+2+4 = 38
ΣIxIt = -4-1-9-4-16-1-9-4-1 = -49
ΣIyIt = 2+3+6+8+4+3+6+2+4 = 38
```

### Step 3: Solve System
```
[49  38] [u]   [49]
[38  61] [v] = [38]

Using Cramer's rule:
det = 49×61 - 38×38 = 2989 - 1444 = 1545

u = (49×61 - 38×38)/1545 = (2989-1444)/1545 = 1545/1545 = 1.0
v = (49×38 - 38×49)/1545 = 0/1545 = 0.0
```

**Result:** Flow vector = (1.0, 0.0) pixels → **Horizontal motion**

---

# Step-by-Step Algorithms

## Algorithm 1: Canny Edge Detection

```python
def canny_edge_detection(image, sigma=1.0, t_high=0.2, t_low=0.1):
    """
    Complete Canny edge detection algorithm
    """
    # Step 1: Gaussian smoothing
    gaussian_kernel = create_gaussian_kernel(sigma)
    smoothed = convolve2d(image, gaussian_kernel)
    
    # Step 2: Gradient computation
    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    grad_x = convolve2d(smoothed, sobel_x)
    grad_y = convolve2d(smoothed, sobel_y)
    
    magnitude = sqrt(grad_x² + grad_y²)
    direction = arctan2(grad_y, grad_x)
    
    # Step 3: Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, direction)
    
    # Step 4: Double thresholding  
    strong_edges = suppressed > t_high
    weak_edges = (suppressed >= t_low) & (suppressed <= t_high)
    
    # Step 5: Edge tracking by hysteresis
    final_edges = hysteresis_tracking(strong_edges, weak_edges)
    
    return final_edges

def non_maximum_suppression(magnitude, direction):
    """
    Thin edges to single pixel width
    """
    result = zeros_like(magnitude)
    
    # Quantize directions to 0°, 45°, 90°, 135°
    angle = direction * 180 / pi
    angle[angle < 0] += 180
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Determine neighbors based on gradient direction
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                neighbors = [magnitude[i,j-1], magnitude[i,j+1]]  # Horizontal
            elif 22.5 <= angle[i,j] < 67.5:
                neighbors = [magnitude[i-1,j-1], magnitude[i+1,j+1]]  # Diagonal /
            elif 67.5 <= angle[i,j] < 112.5:
                neighbors = [magnitude[i-1,j], magnitude[i+1,j]]  # Vertical
            else:  # 112.5 <= angle[i,j] < 157.5
                neighbors = [magnitude[i-1,j+1], magnitude[i+1,j-1]]  # Diagonal \
            
            # Keep pixel if it's maximum along gradient direction
            if magnitude[i,j] >= max(neighbors):
                result[i,j] = magnitude[i,j]
    
    return result
```

## Algorithm 2: Harris Corner Detection

```python
def harris_corner_detection(image, k=0.05, window_size=5, threshold=0.01):
    """
    Complete Harris corner detection
    """
    # Step 1: Gaussian smoothing
    smoothed = gaussian_blur(image, sigma=1.0)
    
    # Step 2: Compute gradients
    grad_x = convolve2d(smoothed, sobel_x_kernel)
    grad_y = convolve2d(smoothed, sobel_y_kernel)
    
    # Step 3: Compute structure matrix components
    Ixx = grad_x * grad_x
    Iyy = grad_y * grad_y  
    Ixy = grad_x * grad_y
    
    # Step 4: Apply Gaussian weighting
    gaussian_window = create_gaussian_kernel(window_size, sigma=1.5)
    
    Sxx = convolve2d(Ixx, gaussian_window)
    Syy = convolve2d(Iyy, gaussian_window)
    Sxy = convolve2d(Ixy, gaussian_window)
    
    # Step 5: Compute Harris response
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    harris_response = det_M - k * trace_M * trace_M
    
    # Step 6: Threshold and non-maximum suppression
    corners = harris_response > threshold
    corners = non_maximum_suppression_2d(harris_response, corners)
    
    return corners, harris_response
```

## Algorithm 3: Zhang's Camera Calibration

```python
def zhang_calibration(world_points, image_points_list):
    """
    Zhang's planar calibration method
    """
    homographies = []
    
    # Step 1: Estimate homographies for each view
    for image_points in image_points_list:
        H = estimate_homography_dlt(world_points, image_points)
        homographies.append(H)
    
    # Step 2: Solve for intrinsic parameters
    V = []
    for H in homographies:
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
        
        # Orthogonality constraints
        v12 = create_v_vector(h1, h2)
        v11_minus_v22 = create_v_vector(h1, h1) - create_v_vector(h2, h2)
        
        V.append(v12)
        V.append(v11_minus_v22)
    
    # Solve Vb = 0 for b (parameters of B = K^-T K^-1)
    V = np.array(V)
    _, _, Vt = svd(V)
    b = Vt[-1, :]
    
    # Recover intrinsic matrix K from b
    K = recover_intrinsics_from_b(b)
    
    # Step 3: Compute extrinsics for each view
    extrinsics = []
    for H in homographies:
        R, t = compute_extrinsics(K, H)
        extrinsics.append((R, t))
    
    # Step 4: Refine all parameters (optional bundle adjustment)
    K_refined, extrinsics_refined = bundle_adjustment(
        K, extrinsics, world_points, image_points_list)
    
    return K_refined, extrinsics_refined

def create_v_vector(h1, h2):
    """Create constraint vector for homography columns h1, h2"""
    return np.array([
        h1[0]*h2[0],
        h1[0]*h2[1] + h1[1]*h2[0],
        h1[1]*h2[1],
        h1[2]*h2[0] + h1[0]*h2[2],
        h1[2]*h2[1] + h1[1]*h2[2],
        h1[2]*h2[2]
    ])
```

---

# Quick Reference Cheat Sheet

## 📐 Essential Formulas

### Camera Projection
```
Perspective:     s·p = K[R|t]P
Orthographic:    p = S·P + c  (S=scaling, c=center)
Homography:      p' = H·p     (planar surfaces)
```

### Intrinsic Matrix
```
K = [fx  s  cx]    fx,fy: focal lengths (pixels)
    [ 0 fy  cy]    cx,cy: principal point  
    [ 0  0   1]    s: skew (usually ≈ 0)
```

### Transformations
```
Rotation:    R = [cos θ  -sin θ]
                 [sin θ   cos θ]

Translation: T = [1  0  tx]
                 [0  1  ty] 
                 [0  0   1]

Scaling:     S = [sx  0   0]
                 [ 0 sy   0]
                 [ 0  0   1]
```

### Fourier Transform
```
2D DFT: F[k,l] = Σₘₙ f[m,n]·e^(-j2π(km/M + ln/N))
Convolution ↔ Multiplication: f⊗h ↔ F·H
```

### Edge Detection
```
Gradient magnitude: |∇f| = √(fx² + fy²)
Gradient direction: θ = arctan2(fy, fx)
Laplacian: ∇²f = fxx + fyy
```

### Harris Corners
```
Structure matrix: M = [Σ Ix²   Σ IxIy]
                      [Σ IxIy  Σ Iy² ]

Harris response: R = det(M) - k·trace(M)²
k ∈ [0.04, 0.06]
```

### Optical Flow
```
Constraint equation: Ix·u + Iy·v + It = 0
Lucas-Kanade: [Σ Ix²   Σ IxIy] [u] = [-Σ IxIt]
              [Σ IxIy  Σ Iy² ] [v]   [-Σ IyIt]
```

### Stereo Vision
```
Depth from disparity: Z = f·B/d
Disparity: d = uL - uR
Baseline: B (distance between cameras)
```

## 🎯 Key Parameter Guidelines

### Canny Edge Detection
```
σ (Gaussian):     0.5-2.0 pixels
T_high:          0.1-0.3 × max(gradient)
T_low:           0.4 × T_high
```

### Harris Corners
```
k parameter:     0.04-0.06
Window size:     5×5 to 7×7 pixels
Gaussian σ:      1.0-1.5 pixels
```

### Camera Calibration
```
Minimum images:  3 (practical: 10-20)
Pattern angles:  Vary by 30°+ between views
Coverage:        Fill entire image area
```

## 🔍 Algorithm Checklist

### Canny Steps
1. ✓ Gaussian blur (noise reduction)
2. ✓ Sobel gradients (magnitude + direction)  
3. ✓ Non-maximum suppression (thin edges)
4. ✓ Double threshold (strong/weak classification)
5. ✓ Hysteresis tracking (connect weak to strong)

### Harris Steps  
1. ✓ Gaussian smoothing
2. ✓ Compute Ix, Iy gradients
3. ✓ Form structure matrix M (Gaussian weighted)
4. ✓ Calculate R = det(M) - k·trace(M)²
5. ✓ Threshold and non-maximum suppression

### Zhang Calibration Steps
1. ✓ Capture multiple checkerboard views
2. ✓ Detect corners with subpixel accuracy
3. ✓ Estimate homography for each view (DLT)
4. ✓ Solve linear system for intrinsics
5. ✓ Compute extrinsics for each view
6. ✓ Bundle adjustment refinement

## 📊 Important Comparisons

| Aspect | Perspective | Orthographic |
|--------|-------------|--------------|
| Rays | Converge at center | Parallel |
| Depth effect | Size ∝ 1/Z | Size constant |
| FOV | Wide angles | Narrow/telephoto |
| Distortion | Foreshortening | None |

| Reflection | Lambertian | Specular |
|------------|------------|----------|
| Dependence | Surface normal | View direction |
| Formula | I ∝ n·l | I ∝ (r·v)ᵐ |
| Appearance | Matte/diffuse | Shiny/glossy |
| Viewing | Same from all angles | Changes with viewpoint |

| Filter | Sobel | Scharr | Prewitt |
|--------|-------|--------|---------|
| Smoothing | Moderate | Better | Less |
| Isotropy | Good | Excellent | Poor |
| Noise | Moderate | Low | Higher |

---

## 💡 Exam Tips

### Formula Memorization
- **Always write homogeneous coordinates** for transformations
- **Remember the k parameter range** for Harris (0.04-0.06)  
- **Canny threshold ratio**: T_low = 0.4 × T_high
- **RGB to grayscale**: 0.299R + 0.587G + 0.114B

### Problem-Solving Strategy
1. **Identify the problem type** (calibration, feature detection, etc.)
2. **List known parameters** and what needs to be found
3. **Choose appropriate method** (Zhang for calibration, Lucas-Kanade for flow)
4. **Show mathematical steps** clearly
5. **Check units and reasonableness** of final answer

### Common Pitfalls
- Forgetting to normalize homogeneous coordinates
- Using wrong coordinate system (image vs. camera vs. world)
- Incorrect matrix dimensions in multiplications
- Not applying non-maximum suppression after thresholding

---

**Good luck with your exam! 🎓**

*This study guide covers all major topics from Units 1-3. Practice with the numerical examples and make sure you understand the underlying concepts, not just memorize formulas.*
