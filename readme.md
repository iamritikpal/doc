# 📚 Computer Vision Exam Notes (Simple & Easy)
## Units 1-3 | Quick Study Guide

**Perfect for Last-Minute Revision! 🎯**

---

# 🚀 Quick Navigation
- [Unit 1: Cameras & Images](#unit-1-cameras--images) 
- [Unit 2: Filters & Features](#unit-2-filters--features)
- [Unit 3: Motion & 3D](#unit-3-motion--3d)
- [Important Formulas](#important-formulas)
- [Solved Examples](#solved-examples)
- [Exam Tips](#exam-tips)

---

# Unit 1: Cameras & Images

## 📷 How Cameras Work

### Simple Camera Model
Think of a camera like your eye:

```
Real World Object → Camera → Image on Screen

     🏠 House
      |
      | (light rays)
      ↓
   📷 Camera
      |
      ↓
   📱 Photo of house
```

**Key Point:** 3D world becomes 2D image!

### Two Main Types:

#### 1. Perspective Camera (Normal Camera)
- **What it does:** Far objects look smaller
- **Formula:** `x = f × (X/Z)`, `y = f × (Y/Z)`
- **Remember:** Divide by Z (depth)!

```
Far objects: Small in image  📏
Near objects: Big in image   📏📏📏
```

#### 2. Orthographic Camera (Special Case)
- **What it does:** All objects same size (no depth effect)
- **Formula:** `x = X`, `y = Y` (just copy coordinates)
- **When used:** Very far objects, satellites

## 🎛️ Camera Settings (Intrinsic Parameters)

**Camera Matrix K:**
```
K = [fx  0  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```

**What each means:**
- **fx, fy:** How "zoomed in" the camera is (bigger = more zoom)
- **cx, cy:** Center of the image (usually middle)
- **0, 0, 1:** Always these values (don't change)

## 🔄 Moving & Rotating Images

### Basic Transformations (2D):

#### 1. Move Image (Translation)
```
T = [1  0  tx]  ← tx = move right/left
    [0  1  ty]  ← ty = move up/down  
    [0  0   1]
```

#### 2. Rotate Image
```
R = [cos θ  -sin θ  0]
    [sin θ   cos θ  0]
    [ 0       0     1]
```
**Memory trick:** cos on diagonal, -sin and +sin

#### 3. Make Bigger/Smaller (Scaling)
```
S = [sx  0   0]  ← sx = width scale
    [ 0 sy   0]  ← sy = height scale
    [ 0  0   1]
```

#### 4. Skew/Shear (Slant)
```
H = [1  k  0]  ← k = how much to slant
    [0  1  0]
    [0  0  1]
```

## 🌈 Colors & Light

### RGB Colors
```
Red   = [255,   0,   0]
Green = [  0, 255,   0]  
Blue  = [  0,   0, 255]
White = [255, 255, 255]
Black = [  0,   0,   0]
```

### Convert RGB to Grayscale
**Formula:** `Gray = 0.3×R + 0.6×G + 0.1×B`
**Memory trick:** Green is brightest, Red medium, Blue darkest

### Light Reflection

#### Lambertian (Matte surfaces)
- **Formula:** `I = k × max(0, n·l)`
- **What it means:** Brightness depends on angle of light
- **Example:** Paper, chalk, matte paint

#### Specular (Shiny surfaces)  
- **Formula:** `I = k × max(0, r·v)^m`
- **What it means:** Shiny spot depends on viewing angle
- **Example:** Mirror, metal, water

## 📐 Camera Calibration (Zhang's Method)

**Goal:** Find camera settings from checkerboard photos

### Simple Steps:
1. **Take photos** of checkerboard from different angles
2. **Find corners** in each photo
3. **Calculate homography** (transformation) for each photo
4. **Solve equations** to get camera matrix K
5. **Refine** with optimization

**Key Formula:** `H = K × [r1 r2 t]`
- H = homography (what we measure)
- K = camera matrix (what we want)
- r1, r2, t = rotation and translation

---

# Unit 2: Filters & Features

## 🌊 Fourier Transform (Frequency Analysis)

### Simple Idea
**Every image = combination of sine waves**

```
Low Frequency  = Smooth parts    ∼∼∼∼
High Frequency = Edges & details ∿∿∿∿
```

**2D Fourier Formula:**
```
F[k,l] = Σ f[m,n] × e^(-j2π(km/M + ln/N))
```

**Don't panic!** Just remember:
- **F[0,0]** = average brightness
- **Corners of F** = high frequencies (edges)
- **Center of F** = low frequencies (smooth areas)

### Convolution Theorem
**Key Rule:** `Multiply in frequency = Convolve in space`

## 🔍 Image Filters

### 1. Gaussian Filter (Blur)
**Purpose:** Remove noise, make image smooth

```
Gaussian Kernel (3×3):
1/16 × [1  2  1]
       [2  4  2]  
       [1  2  1]
```

**Parameter σ (sigma):**
- Small σ (0.5-1): Little blur
- Large σ (2-5): Lots of blur

### 2. Sobel Filter (Find Edges)
**Purpose:** Detect edges in X and Y directions

```
Sobel X:           Sobel Y:
[-1  0  1]        [-1 -2 -1]
[-2  0  2]        [ 0  0  0]
[-1  0  1]        [ 1  2  1]
```

**How to use:**
1. Apply both filters
2. Magnitude = √(Gx² + Gy²)
3. Direction = atan2(Gy, Gx)

### 3. Laplacian (Second Derivative)
**Purpose:** Find rapid changes (edges)

```
Laplacian:
[ 0 -1  0]
[-1  4 -1]
[ 0 -1  0]
```

## 🎯 Canny Edge Detection (5 Steps)

**Goal:** Find thin, clean edges

### Step 1: Smooth with Gaussian
- Remove noise first
- Use σ = 1.0 typically

### Step 2: Find Gradients  
- Use Sobel filters
- Get magnitude and direction

### Step 3: Non-Maximum Suppression
- **What:** Make thick edges thin
- **How:** Keep only pixels that are maximum in gradient direction

### Step 4: Double Thresholding
- **High threshold:** Definitely edges
- **Low threshold:** Maybe edges  
- **Rule:** T_low = 0.4 × T_high

### Step 5: Hysteresis Tracking
- **What:** Connect weak edges to strong edges
- **How:** Follow chains of connected pixels

```
Pipeline: Image → Blur → Gradients → Thin → Threshold → Connect → Edges
```

**Parameter Tips:**
- σ too small → noisy edges
- σ too large → missing edges  
- T_high too low → too many edges
- T_high too high → missing edges

## 🔲 Harris Corner Detection

**Goal:** Find corners (points where edges meet)

### Simple Idea:
```
Flat area:    No change in any direction     ....
Edge:         Change in one direction        ||||
Corner:       Change in two directions       ┼┼┼┼
```

### Math (Don't Memorize, Understand):

1. **Structure Matrix M:**
```
M = [Ix²   IxIy]  ← Ix, Iy are gradients
    [IxIy  Iy² ]
```

2. **Harris Response:**
```
R = det(M) - k × trace(M)²
where k = 0.05 (always use this value)
```

3. **Interpretation:**
- R > 0: Corner ✓
- R < 0: Edge
- R ≈ 0: Flat area

### Algorithm Steps:
1. Smooth image
2. Compute gradients Ix, Iy
3. Compute M for each pixel (with Gaussian window)
4. Compute R = det(M) - 0.05 × trace(M)²
5. Threshold: keep R > some value
6. Non-maximum suppression

## 🌾 Texture Analysis

### GLCM (Gray Level Co-occurrence Matrix)
**Simple idea:** Look at pairs of pixels

**Common features:**
- **Contrast:** How different are neighboring pixels?
- **Energy:** How uniform is the texture?
- **Homogeneity:** How smooth is the texture?

### LBP (Local Binary Patterns)
**Simple idea:** Compare each pixel with its neighbors

```
Example:
6  5  2     1  1  0     Binary: 11000111 = 199
7  6  1  →  1  ·  0  →  
9  8  7     1  1  1     
```

---

# Unit 3: Motion & 3D

## 👁️ Optical Flow (Motion Detection)

### Basic Idea
**Track how pixels move between frames**

```
Frame 1:  ●        Frame 2:    ●     Motion: →
          ●                      ●             →
```

### Brightness Constancy
**Assumption:** Pixel brightness stays same as it moves
**Formula:** `I(x,y,t) = I(x+u, y+v, t+1)`

### Optical Flow Constraint Equation (OFCE)
**The most important equation:**
```
Ix × u + Iy × v + It = 0
```

**What it means:**
- Ix, Iy: spatial gradients (how image changes in x,y)
- It: temporal gradient (how image changes in time)
- u, v: motion we want to find

**Problem:** 1 equation, 2 unknowns (u,v)!

### Lucas-Kanade Solution
**Idea:** Assume motion is same in small window

**Matrix form:**
```
[Ix₁  Iy₁]     [-It₁]
[Ix₂  Iy₂] [u] [-It₂]
[... ...] [v] = [...]
[IxN  IyN]     [-ItN]
```

**Solution:**
```
[u] = (AᵀA)⁻¹ × Aᵀb
[v]
```

**When it works:** Corners, textured areas
**When it fails:** Uniform areas, large motions

### Pyramidal Lucas-Kanade
**Problem:** Large motions break the method
**Solution:** Use image pyramids (coarse to fine)

```
Level 2: [▓▓]     ← Find large motion
Level 1: [▓▓▓▓]   ← Refine motion  
Level 0: [████████] ← Final refinement
```

## 👀 Stereo Vision (3D from Two Cameras)

### Basic Setup
```
Left Camera    Right Camera
     |              |
     |              |
 [Left Image]  [Right Image]
```

### Key Concepts

#### Disparity
**Definition:** Horizontal difference between same point in two images
```
d = uL - uR
```

#### Depth from Disparity
**Magic formula:**
```
Z = f × B / d
```
**Where:**
- Z = depth (what we want)
- f = focal length
- B = baseline (distance between cameras)  
- d = disparity (what we measure)

**Key insight:** 
- Large disparity → Close object
- Small disparity → Far object

#### Stereo Rectification
**Goal:** Make epipolar lines horizontal
**Why:** Reduces 2D search to 1D search

```
Before:     After:
  ╱╲         ——
 ╱  ╲        ——  ← Horizontal lines
╱____╲       ——
```

### Correspondence Problem
**Challenge:** Which pixel in left image matches which pixel in right image?

**Matching costs:**
- **SAD:** Sum of Absolute Differences
- **SSD:** Sum of Squared Differences  
- **NCC:** Normalized Cross Correlation

## 🏗️ Structure from Motion

### Fundamental Matrix F
**Relates corresponding points in two views:**
```
p₂ᵀ × F × p₁ = 0
```

### Essential Matrix E (for calibrated cameras)
```
E = [t]× × R
```

### 8-Point Algorithm
**Find F from ≥8 point correspondences**
- Set up linear system Af = 0
- Solve using SVD

---

# Important Formulas

## 🔢 Must-Know Equations

### Camera Projection
```
Perspective: [u,v,1] = K[R|t][X,Y,Z,1]
Homography:  p' = H × p
```

### Intrinsic Matrix
```
K = [fx  0  cx]
    [ 0 fy  cy]  
    [ 0  0   1]
```

### Transformations
```
Translation: [1 0 tx; 0 1 ty; 0 0 1]
Rotation:    [cos θ -sin θ 0; sin θ cos θ 0; 0 0 1]
Scaling:     [sx 0 0; 0 sy 0; 0 0 1]
```

### Edge Detection
```
Gradient magnitude: |∇f| = √(fx² + fy²)
Gradient direction: θ = atan2(fy, fx)
```

### Harris Corners
```
M = [Ix²   IxIy]
    [IxIy  Iy² ]
R = det(M) - k×trace(M)²  (k = 0.05)
```

### Optical Flow
```
OFCE: Ix×u + Iy×v + It = 0
Lucas-Kanade: [ΣIx² ΣIxIy; ΣIxIy ΣIy²][u;v] = [-ΣIxIt; -ΣIyIt]
```

### Stereo Depth
```
Z = f × B / d
d = uL - uR
```

### Color Conversion
```
Gray = 0.299×R + 0.587×G + 0.114×B
```

---

# Solved Examples

## Example 1: Camera Calibration (Simple)

**Given:** Checkerboard with 20mm squares, 4 corner points

**World points (mm):** (0,0), (60,0), (60,40), (0,40)
**Image points (px):** (512,290), (900,300), (880,520), (510,500)

**Steps:**
1. Find homography H using DLT
2. Extract camera parameters from H
3. Get focal lengths fx ≈ 1180px, fy ≈ 1170px

**Answer:** 
```
K = [1180   0  640]
    [   0 1170  360]
    [   0   0    1]
```

## Example 2: Harris Corner (Simple)

**Given:** 3×3 patch with gradients
```
Ix = [1  2  1]    Iy = [1  0 -1]
     [0  0  0]         [2  0 -2]
     [-1-2 -1]         [1  0 -1]
```

**Solution:**
1. ΣIx² = 12, ΣIy² = 12, ΣIxIy = 0
2. M = [12 0; 0 12]
3. R = det(M) - 0.05×trace(M)² = 144 - 0.05×576 = 115.2

**Answer:** R = 115.2 > 0 → **Strong Corner!**

## Example 3: Optical Flow (Simple)

**Given:** Window with gradients and temporal differences

**Solution:**
1. Form matrix AᵀA and vector Aᵀb
2. Solve [u;v] = (AᵀA)⁻¹ × Aᵀb
3. Get motion vector (u,v)

---

# Exam Tips

## 🎯 What to Focus On

### High-Priority Topics:
1. **Camera calibration** (Zhang's method)
2. **Canny edge detection** (5 steps)
3. **Harris corner detection** (formula & interpretation)
4. **Optical flow** (OFCE & Lucas-Kanade)
5. **Linear transformations** (rotation, translation, scaling)
6. **Stereo vision** (depth from disparity)

### Medium-Priority Topics:
1. Fourier transform properties
2. Different filters (Gaussian, Sobel, Laplacian)
3. Color spaces (RGB, HSV, YCbCr)
4. Texture analysis (GLCM, LBP)

## 📝 Problem-Solving Strategy

### For Any Problem:
1. **Read carefully** - What is given? What to find?
2. **Identify type** - Calibration? Edge detection? Corner detection?
3. **Choose method** - Zhang for calibration, Canny for edges, etc.
4. **Write formula** - Show the key equation
5. **Substitute values** - Plug in numbers
6. **Calculate step by step** - Show your work
7. **Check answer** - Does it make sense?

### Common Mistakes to Avoid:
- Forgetting to normalize homogeneous coordinates
- Using wrong coordinate system
- Matrix multiplication errors
- Not applying non-maximum suppression
- Wrong parameter values (k=0.05 for Harris, T_low=0.4×T_high for Canny)

## 🧠 Memory Tricks

### Parameter Values:
- **Harris k:** Always 0.05 (between 0.04-0.06)
- **Canny ratio:** T_low = 0.4 × T_high
- **RGB to Gray:** 0.3R + 0.6G + 0.1B (Green is brightest)

### Matrix Patterns:
- **Rotation:** cos on diagonal, -sin and +sin
- **Translation:** 1's on diagonal, tx and ty in last column
- **Scaling:** sx and sy on diagonal

### Algorithm Steps:
- **Canny:** Blur → Gradient → Thin → Threshold → Connect
- **Harris:** Smooth → Gradient → Structure Matrix → Response → Threshold
- **Zhang:** Photos → Corners → Homography → Solve → Refine

## ⏰ Time Management

### For 3-Hour Exam:
- **Theory questions:** 5-10 minutes each
- **Numerical problems:** 15-25 minutes each
- **Algorithm questions:** 10-15 minutes each
- **Keep 30 minutes** for review

### Quick Review Checklist:
- [ ] All formulas written correctly?
- [ ] Units consistent throughout?
- [ ] Matrix dimensions match?
- [ ] Final answers clearly marked?
- [ ] Reasonable values? (focal length ~1000px, not 10 or 100000)

---

## 🎓 Final Advice

1. **Practice numerical examples** - Don't just read, solve!
2. **Understand concepts** - Don't just memorize formulas
3. **Draw diagrams** - Visual understanding helps
4. **Time yourself** - Practice under exam conditions
5. **Review mistakes** - Learn from errors

**Remember:** This is just image processing and geometry. You can do it! 💪

**Good luck! 🍀**
