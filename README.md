# LaunchCAD - Isometric block-based images to CAD
## Inspiration
Computer-Aided Design, or CAD, is integral for planning many engineering projects. However, for many new engineers, having to learn the software and concepts behind CAD to design the elements of their project seems daunting. We asked ourselves, “How can we lower the barrier of entry so beginners can start CADing without tons of technical experience?” If only there were a way to simplify the process so engineers can start prototyping right away… *drum roll* … introducing LaunchCAD, a way to CAD that is as simple as putting pen to paper, a launchpad for every engineer’s journey.

## What it does
LaunchCAD converts two-dimensional isometric drawings into three-dimensional CAD models. Using a mobile device, users can scan an isometric drawing and upload it directly onto a computer where the image will be converted into a .stl file. These files can then be used however the user wants to, whether for additional tweaking, 3D printing, or other such projects.

## Software Design
The primary components of this software are to:
- Read in files from base64 format and convert such files to the appropriate `numpy` and `cv2` data structures
- Perform template matching and coordinate normalization on our image to locate cubes in isometric drawing
- Convert `numpy` meshes/arrays into `.stl` files
- Expose `IsoParser` functionality through `tornado` api

(Updated 2023) In an effort of simplicity and readability, we have modeled our interactions as such:


## Algorithm Design
The premise of this algorithm is to match the 3 possible faces of each cube to the isometric drawing and somehow map out where, in the 3 dimensions, these cubes lie relative to the other cubes. The current release is able to do this for 1 type of face on two dimensions.

### Step 1 - Template matching

### Step 2 - Coordinate normalization

### Step 3 - Putting it all together

# Demonstration
Please check out this video in which we demo the initial release: [LaunchCAD - HackGT8](https://www.youtube.com/watch?v=GyFxF24hYEk)
