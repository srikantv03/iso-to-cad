## Iso-To-Cad
# Inspiration
CAD, also known as Computer-Aided Design, is an integral process towards the planning portion of many engineering projects. However, for many new engineers, having to learn the software and concepts behind CAD just to design the elements of their project seems daunting. We asked ourselves, “How can we lower the barrier of entry so beginners can start CADing without tons of technical experience?” If only there was a way to simplify the process so engineers can start prototyping right away… drum roll … introducing LaunchCAD, a way to CAD that is as simple as putting pen to paper, a launchpad for every engineer’s journey.

# What it does
LaunchCAD converts two-dimensional isometric drawings into three-dimensional CAD models. Using a mobile device, users can scan an isometric drawing and upload it directly onto a computer where the image will be converted into a .stl file. These files can then be used however the user wants to, whether for additional tweaking, 3D printing, or other such projects.

# How we built it
Using open-source libraries and our own code, we were able to apply neural networks, optimization algorithms, and different mathematical models to create our application. The primary conversion from the isometric drawing to tangible data was through the use of OpenCV’s template matching CNN (convolutional neural network). On top of the template matching, we used trigonometric calculations to straighten out the isometric image into planes of 2d matrices. Finally, using statistical analysis and NumPy meshes, we were able to determine the relative location of each cube and assemble a CAD model. The frontend of our application was built using React Native and the backend was built using Tornado.

# Challenges we ran into
How to detect each cube in the CAD model
Line detection could not find the vertices accurately, causing us to have improperly positioned objects
Template matching would scramble to find matches at very low confidence and cause infinite loops
Murphy’s Law specifically for Android Studio and npm packages (everything that can go wrong will go wrong)
Finding viable camera packages that perform Base64 encoding efficiently and compressing high-density image files
# Accomplishments that we're proud of
Working minimum viable product with a sleek UI design, accurate results, and a promising future for the application and the algorithm
Able to go from isometric drawing to full-fledged CAD drawing, which accomplishes our initial goal to help engineers fast-track to the prototyping stage of their engineering design process
# What's next for LaunchCAD
Account for more complex drawings and shapes (potentially using models scraped on large datasets)
Built-in warp transform for document scanning
Live-time CAD previews
Coloring detection


## Citations


OpenCV
https://github.com/opencv/opencv-python/tree/master/cv2
Numpy
https://github.com/numpy/numpy/tree/main/numpy
Numpy-Stl
https://github.com/WoLpH/numpy-stl/tree/develop/stl
MatPlotLib
https://github.com/matplotlib/matplotlib
Tornado
https://github.com/tornadoweb/tornado/blob/master/tornado/ioloop.py


"@react-navigation/native": "^6.0.6",
    "@react-navigation/native-stack": "^6.2.5",
https://reactnavigation.org/docs/stack-navigator/
   "expo": "~43.0.0",
https://expo.dev/
    "expo-splash-screen": "~0.13.3",
https://docs.expo.dev/versions/latest/sdk/splash-screen/
    "expo-status-bar": "~1.1.0",
https://docs.expo.dev/versions/latest/sdk/status-bar/
    "expo-updates": "~0.10.5",
https://docs.expo.dev/versions/latest/sdk/updates/
    "fs": "0.0.1-security",
https://github.com/nodejs/node/blob/master/doc/api/fs.md
    "halogenium": "^2.3.0",
https://www.npmjs.com/package/halogenium
    "lottie-react-native": "^4.1.3",
https://github.com/lottie-react-native/lottie-react-native
    "npm-react-component-starter": "^1.0.0",
https://www.npmjs.com/package/npm-react-component-starter
    "react": "17.0.1",
https://www.npmjs.com/package/react
