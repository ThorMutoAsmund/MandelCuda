﻿kernel void mandel(global read_only int* message, int N, global read_only int* gradient, int gradientLength, double ymin, double xmin, double width, int maxiter, int colorCycleOffset)
{
    double cy;
    double cx;
    double x;
    double y;
    double xtmp;
    int iter;
    int i = get_global_id(0);
    int j = get_global_id(1);
    cy = ymin + i * width;
    cx = xmin + j * width;

    x = 0;
    y = 0;
    iter = 0;

    while (x * y < 4.0 && iter < maxiter)
    {
        xtmp = x;
        x = x * x - y * y + cx;
        y = 2 * xtmp * y + cy;
        iter++;
    }
    if (iter == maxiter)
    {
        message[i * N + j] = 0;
    }
    else
    {
        message[i * N + j] = gradient[(iter + colorCycleOffset) % gradientLength];
    }
}
