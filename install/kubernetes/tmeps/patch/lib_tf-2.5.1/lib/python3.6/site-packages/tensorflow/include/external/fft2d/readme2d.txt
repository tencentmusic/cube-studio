General Purpose 2D,3D FFT (Fast Fourier Transform) Package

Files
    alloc.c    : 2D-array Allocation
    alloc.h    : 2D-array Allocation
    fft4f2d.c  : 2D FFT Package in C       - Version I (radix 4, 2)
    fft4f2d.f  : 2D FFT Package in Fortran - Version I (radix 4, 2)
    fftsg.c    : 1D FFT Package in C       - Fast Version (Split-Radix)
    fftsg.f    : 1D FFT Package in Fortran - Fast Version (Split-Radix)
    fftsg2d.c  : 2D FFT Package in C       - Version II (Split-Radix)
    fftsg2d.f  : 2D FFT Package in Fortran - Version II (Split-Radix)
    fftsg3d.c  : 3D FFT Package in C       - Version II (Split-Radix)
    fftsg3d.f  : 3D FFT Package in Fortran - Version II (Split-Radix)
    shrtdct.c  : 8x8, 16x16 DCT Package
    sample2d/
        Makefile    : for gcc, cc
        Makefile.f77: for Fortran
        Makefile.pth: Pthread version
        fft4f2dt.c  : Test Program for "fft4f2d.c"
        fft4f2dt.f  : Test Program for "fft4f2d.f"
        fftsg2dt.c  : Test Program for "fftsg2d.c"
        fftsg2dt.f  : Test Program for "fftsg2d.f"
        fftsg3dt.c  : Test Program for "fftsg3d.c"
        fftsg3dt.f  : Test Program for "fftsg3d.f"
        shrtdctt.c  : Test Program for "shrtdct.c"

Difference of Files
    C and Fortran versions are equal and 
    the same routines are in each version.
    ---- Difference between "fft4f2d.*" and "fftsg2d.*" ----
    "fft4f2d.*" are optimized for the old machines that 
    don't have the large size CPU cache.
    "fftsg2d.*", "fftsg3d.*" use 1D FFT routines in "fftsg.*".
    "fftsg2d.*", "fftsg3d.*" are optimized for the machines that 
    have the multi-level (L1,L2,etc) cache.

Routines in the Package
    in fft4f2d.*, fftsg2d.*
        cdft2d: 2-dim Complex Discrete Fourier Transform
        rdft2d: 2-dim Real Discrete Fourier Transform
        ddct2d: 2-dim Discrete Cosine Transform
        ddst2d: 2-dim Discrete Sine Transform
        rdft2dsort: rdft2d input/output ordering (fftsg2d.*)
    in fftsg3d.*
        cdft3d: 3-dim Complex Discrete Fourier Transform
        rdft3d: 3-dim Real Discrete Fourier Transform
        ddct3d: 3-dim Discrete Cosine Transform
        ddst3d: 3-dim Discrete Sine Transform
        rdft3dsort: rdft3d input/output ordering
    in fftsg.*
        cdft: 1-dim Complex Discrete Fourier Transform
        rdft: 1-dim Real Discrete Fourier Transform
        ddct: 1-dim Discrete Cosine Transform
        ddst: 1-dim Discrete Sine Transform
        dfct: 1-dim Real Symmetric DFT
        dfst: 1-dim Real Anti-symmetric DFT
        (these routines are called by fftsg2d.*, fftsg3d.*)
    in shrtdct.c
        ddct8x8s  : Normalized 8x8 DCT
        ddct16x16s: Normalized 16x16 DCT
        (faster than ddct2d())

Usage
    Brief explanations are in block comments of each packages.
    The examples are given in the test programs.

Copyright
    Copyright(C) 1997,2001 Takuya OOURA (email: ooura@kurims.kyoto-u.ac.jp).
    You may use, copy, modify this code for any purpose and 
    without fee. You may distribute this ORIGINAL package.

History
    ...
    Nov. 2001  : Add 3D-FFT routines
    Dec. 2006  : Fix a documentation bug in "fftsg3d.*"
    Dec. 2006  : Fix a minor bug in "fftsg.f"

