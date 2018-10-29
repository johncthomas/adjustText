<img src="https://github.com/Phlya/adjustText/blob/master/adjustText_logo.svg" width="183">

Inspired by **ggrepel** package for R/ggplot2 (https://github.com/slowkow/ggrepel) 
![Alt text](figures/mtcars.gif "Labelled mtcars dataset")

The idea is that often when we want to label multiple points on a graph the text will start heavily overlapping with 
both other labels and data points. This can be a major problem requiring manual solution. However this can be largely 
automatized by smart placing of the labels (difficult) or iterative adjustment of their positions to minimize overlaps 
(relatively easy). This library (well... script) implements the latter option and should be adatable for use
with many plotting packages, including Bokeh. Usage is 
very straightforward with usually pretty good results with no tweaking (most important is to just make text slightly 
smaller than default and maybe the figure a little larger). However the algorithm itself is highly configurable for 
complicated plots.

This version is adapted from the original https://github.com/Phlya/adjustText/ for use with other plotting packages 
(specifically Bokeh, but it doesn't directly interact with external packages).
A limitation not present in the original is that the user must use a fixed width font and supply the pixel width and 
height, as well as the plot dimensions (pixel and data). If you're working with Matplotlib, the original is simpler to
use.

For the latest version from github:
```
pip install https://github.com/Phlya/adjustText/archive/master.zip
```

See [wiki] for some basic introduction, and more advanced usage examples [here].

[wiki]: https://github.com/Phlya/adjustText/wiki
[here]: https://github.com/Phlya/adjustText/blob/master/docs/source/Examples.ipynb
