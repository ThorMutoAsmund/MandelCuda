kernel void mandel(global read_only int* message, int N, float ymin, float xmin, float width, int maxiter)
{
    int messageSize = N * N;
    float cy;
    float cx;
    float x;
    float y;
    float xtmp;
    int iter;
    int i = get_global_id(0);
    //int io = get_global_offset(0); 
    //for (int i = 0; i < N; i++)
    //{
    cy = ymin + i * width;
    int j = get_global_id(1);
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
        message[i * N + j] = iter;
    //}
    //}
}
