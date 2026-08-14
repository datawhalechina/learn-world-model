---
title: "Optional Frontier Survey: Spatial 3D/4D World Models"
description: Predicting in 3D/4D scene space instead of pixels or a flat latent vector, using camera geometry, depth, NeRF, 3D Gaussian Splatting, and scene flow.
lecture: 3
---

# Optional Frontier Survey: Spatial 3D/4D World Models

## Architecture Six: Spatial 3D/4D World Models

**Representative systems**: NeRF (2020), 3D Gaussian Splatting (2023), 4D Gaussian Splatting and dynamic scene extensions (2024+)

JEPA changed *what* is predicted, from pixels to semantic representations. Spatial 3D/4D models change *where* prediction happens: instead of a flat latent vector or a 2D pixel grid, the state is an explicit representation of 3D scene geometry, and prediction operates over that geometry directly.

### Why a Flat Latent Vector Can Lose the Wrong Things

RSSM, Transformer, and diffusion backbones covered so far all treat the observation as a 2D image (or a sequence of them) and compress it into a vector or a token grid. This works well for control tasks defined by 2D visual patterns, such as Atari or DMControl. It works less well for a specific failure mode: **object permanence** (an object should still be understood to exist and hold its position when temporarily occluded, out of frame, or viewed from a new angle) and **multi-view consistency** (the same physical scene, observed from different camera positions, should decode to geometrically consistent structure). A 2D latent vector has no explicit notion of "where things are in 3D space," so nothing forces the model's predictions to respect the physical constraint that an occluded object has not vanished.

Spatial world models address this by giving the latent state actual 3D structure.

### Core Mechanism: Representing Geometry Directly

> **📖 Camera geometry basics**: A **camera pose** specifies a camera's position and orientation in 3D space, usually as a rotation matrix and a translation vector. **Depth** is the distance from the camera to each point in the scene, one value per pixel. Given a pixel's depth and the camera pose, that pixel can be projected back into a 3D point. A **point cloud** is a set of such 3D points, with no connectivity between them. These quantities let a system convert between "what does the scene look like from this viewpoint" and "where is this point in the world," which a flat latent vector cannot do without learning the relationship implicitly.

Two representative approaches make 3D structure explicit and learnable:

**NeRF** (Neural Radiance Fields, [Mildenhall et al., 2020](https://arxiv.org/abs/2003.08934)) represents a scene as a continuous function: given a 3D point and a viewing direction, a neural network outputs a color and a density. Rendering an image from a given camera pose means casting a ray through each pixel, sampling points along the ray, and integrating their predicted color and density, a process called **volume rendering**. The scene representation $\Theta$ is the network's weights, and the render is $I = \text{Renderer}(\Theta, c)$ where $c$ is the camera pose, the same formula introduced in L01's nine-grid table. NeRF is expensive: rendering one image requires many network evaluations per ray, across many rays.

**3D Gaussian Splatting** ([Kerbl et al., 2023](https://arxiv.org/abs/2308.04079)) replaces the implicit neural function with an explicit set of 3D Gaussians, each with a position, covariance (shape and orientation), color, and opacity. Rendering projects these Gaussians onto the image plane and blends them, which is far cheaper than volume rendering and enables real-time rendering rates. Because the Gaussians are an explicit, editable set rather than opaque network weights, individual objects can in principle be manipulated or tracked across time.

### From 3D to 4D: Adding Time

A static NeRF or a static Gaussian scene answers "what does this scene look like," which is an L1 compression capability in the taxonomy from L01, not yet a dynamics model. **4D extensions** add a time or state dimension: each Gaussian (or the radiance field) is allowed to deform over time, conditioned on an observed trajectory or, in the action-conditioned case, on an agent's action. This produces **scene flow**, the 3D-space analogue of optical flow: a per-point velocity field describing how each part of the scene moves between frames. Predicting scene flow forward in time is a 3D/4D world model's version of the dynamics prediction covered in L02, $z_{t+1} = \text{Predictor}(z_{\le t})$, except the representation being rolled forward is explicit 3D geometry rather than an abstract vector.

> **📖 Object permanence via multi-view consistency**: A 3D representation is naturally shared across viewpoints, since it is a single underlying geometry that different camera poses simply render differently. If a model maintains one consistent 3D/4D state rather than predicting each frame independently, an object that a moving camera temporarily loses from view does not need to be "re-discovered" when it reappears. This is a structural advantage over frame-by-frame 2D prediction, which has no mechanism forcing consistency between a frame before and after an occlusion (the same object-persistence weakness noted for Diamond's frame-by-frame diffusion process in L03 Backbone Selection applies to any 2D-only predictor).

**Learning paradigm**: primarily observation-based. Most NeRF and 3DGS training reconstructs a static or passively-observed dynamic scene from multi-view images or video, with no action label required. Action-conditioned 4D world models exist in robotics and driving research (predicting how a scene deforms given a robot's or vehicle's action) but are less mature than the 2D action-conditioned backbones covered in Backbone Selection.

**Applicable scenarios**: autonomous driving perception and forecasting (bird's-eye-view occupancy prediction is a coarse, voxelized cousin of full 4D scene modeling), robotic manipulation where precise 3D object pose matters, and any setting where multi-view consistency or object permanence is a first-class requirement rather than an incidental benefit.

**Limitations**: representing and rendering explicit 3D geometry costs more compute and memory than a flat latent vector, and training typically requires multi-view data (multiple camera angles of the same scene) or precise camera pose annotations, both harder to collect at scale than single-view video. Action-conditioned 4D dynamics prediction, as opposed to passive reconstruction, remains an active research problem rather than a solved component.

## Next: Genie

Spatial 3D/4D models change the representation's geometry while keeping training mostly observation-based. The next page returns to the 2D setting and asks a different question: can a model discover *actions themselves* purely from unlabeled video, with no 3D structure and no action labels at all?

## Further Reading

- [Mildenhall et al. (2020): NeRF](https://arxiv.org/abs/2003.08934): representing scenes as a continuous volumetric radiance field, rendered by ray casting and integration.
- [Kerbl et al. (2023): 3D Gaussian Splatting](https://arxiv.org/abs/2308.04079): an explicit, real-time-renderable alternative to NeRF using a set of anisotropic Gaussians.
- Review the [L1-L5 capability ladder's two-dimensional Object x Capability table](../lecture-01-internal-simulation/02-world-model-taxonomy) for how 3D/4D representations sit alongside features/latents and pixels/video on the object axis.
