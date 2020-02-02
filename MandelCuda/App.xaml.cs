using Cloo;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Input;

// Intorduction
// https://stackoverflow.com/questions/30544082/how-to-pass-large-buffers-to-opencl-devices
// Reading data back
// https://stackoverflow.com/questions/42282548/am-i-reusing-opencl-clooc-objects-correctly
// OpenCL slow: 
// https://sourceforge.net/p/cloo/discussion/1048266/thread/fad3b02e/
// Cloo on SourceForge
// https://sourceforge.net/p/cloo/discussion/1048265/
// Explanation of work items and work groups
// https://downloads.ti.com/mctools/esd/docs/opencl/execution/kernels-workgroups-workitems.html

namespace MandelCuda
{
    class Program
    {
        static int MaxIter = 512;
        static int N = 1024;

        static WriteableBitmap writeableBitmap;
        static Window w;
        static Image i;

        static float ymin;
        static float xmin;
        static float width; 
        static int[] output;

        static int[] colorGradient;
        static int colorGradientLength;

        static float mouseymin;
        static float mousexmin;
        static Point mouseOrig;
        static bool moving;

        [STAThread]
        static void Main(string[] args)
        {
            i = new Image();
            RenderOptions.SetBitmapScalingMode(i, BitmapScalingMode.NearestNeighbor);
            RenderOptions.SetEdgeMode(i, EdgeMode.Aliased);

            w = new Window()
            {
                Width = N,
                Height = N,
                Title = "MandelCuda"
            };
            w.Content = i;
            w.Show();

            writeableBitmap = new WriteableBitmap(
                (int)w.ActualWidth,
                (int)w.ActualHeight,
                96,
                96,
                PixelFormats.Bgr32,
                null);

            i.Source = writeableBitmap;

            i.Stretch = Stretch.None;
            i.HorizontalAlignment = HorizontalAlignment.Left;
            i.VerticalAlignment = VerticalAlignment.Top;

            i.MouseMove += I_MouseMove;
            i.MouseLeftButtonDown += (_, e) => 
            { 
                mouseOrig = e.GetPosition(w); 
                mouseymin = ymin; 
                mousexmin = xmin; 
                moving = true; 
            };
            i.MouseLeftButtonUp += (_, __) => moving = false;

            //i.MouseRightButtonDown +=
            //    new MouseButtonEventHandler(i_MouseRightButtonDown);

            SetColorGradient(new Color[] { Colors.Blue, Colors.White, Colors.Red }, 32);

            w.MouseWheel += W_MouseWheel;

            ymin = -2f;
            xmin = -2f;
            width = 4f;
            output = new int[N * N];

            SetupCUDA(output);
            GenerateMandelBrot(ymin, xmin, width / N, output);
            WriteToBitmap(writeableBitmap);
            
            Application app = new Application();
            app.Run();
        }

        private static void I_MouseMove(object sender, MouseEventArgs e)
        {
            if (moving)
            {
                var pos = e.GetPosition(w);
                xmin = mousexmin - (float)(pos.X - mouseOrig.X) * (width / N);
                ymin = mouseymin - (float)(pos.Y - mouseOrig.Y) * (width / N);

                GenerateMandelBrot(ymin, xmin, width / N, output);
                WriteToBitmap(writeableBitmap);
            }
        }

        private static void W_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            // e.Delta is a multiplum of 120
            if (e.Delta == 0)
            {
                return;
            }

            var pos = e.GetPosition(w);
            const float minm = 0.9f;
            float m = (float)Math.Pow(minm, e.Delta / 120);

            float y = (float)(pos.Y / N);
            float x = (float)(pos.X / N);

            float ycenter = ymin + width * y;
            float xcenter = xmin + width * x;

            width = width * m;
            ymin = ycenter - width*y;
            xmin = xcenter - width*x;

            GenerateMandelBrot(ymin, xmin, width / N, output);
            WriteToBitmap(writeableBitmap);
        }

        private static void SetColorGradient(Color[] colors, int valuesPerColor)
        {
            colorGradientLength = colors.Length * valuesPerColor;
            colorGradient = new int[colorGradientLength];
            for (int c = 0; c < colors.Length; ++c)
            {
                for (int v = 0; v < valuesPerColor; ++v)
                {
                    var c0 = colors[c];
                    var c1 = colors[(c+1) % colors.Length];

                    var d0 = (valuesPerColor-v) / (float)valuesPerColor;
                    var d1 = v / (float)valuesPerColor;
                    colorGradient[c*valuesPerColor + v] = ((int)(c0.R*d0+c1.R*d1)) << 16
                        | ((int)(c0.G*d0 + c1.G * d1)) << 8
                        | (int)(c0.B * d0 + c1.B*d1);

                }
            }
        }

        static void WriteToBitmap(WriteableBitmap bmp)
        {
            try
            {
                // Reserve the back buffer for updates.
                bmp.Lock();

                unsafe
                {
                    // Get a pointer to the back buffer.
                    //b = buffer + i * writeableBitmap.BackBufferStride 
                    IntPtr buffer = writeableBitmap.BackBuffer;

                    for (int i = 0; i < bmp.Height * bmp.Width; ++i)
                    {
                        *((int*)buffer) = output[i] == MaxIter ? 0 : colorGradient[output[i] % colorGradientLength];

                        buffer += 4;
                    }
                }

                // Specify the area of the bitmap that changed.
                bmp.AddDirtyRect(new Int32Rect(0, 0, (int)bmp.Width, (int)bmp.Height));
            }
            finally
            {
                // Release the back buffer and make it available for display.
                bmp.Unlock();
            }
        }

        static ComputePlatform platform;
        static ComputeCommandQueue queue;
        static ComputeContext context;
        static ComputeProgram program;
        static ComputeKernel kernel;
        static ComputeBuffer<int> messageBuffer;
        static int messageSize;

        // 26 ms 4096x4096@512 iter with 1024 cores
        static void SetupCUDA(int[] message)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            // pick first platform
            platform = ComputePlatform.Platforms[0];

            // create context with all gpu devices
            context = new ComputeContext(ComputeDeviceTypes.Gpu,
                new ComputeContextPropertyList(platform), null, IntPtr.Zero);   // LEAK

            // create a command queue with first gpu found
            queue = new ComputeCommandQueue(context,
            context.Devices[0], ComputeCommandQueueFlags.None);

            // load opencl source
            StreamReader streamReader = new StreamReader("Mandel3.cl");
            string clSource = streamReader.ReadToEnd();
            streamReader.Close();

            // create program with opencl source
            program = new ComputeProgram(context, clSource);

            // compile opencl source
            program.Build(null, null, null, IntPtr.Zero);

            // load chosen kernel from program
            kernel = program.CreateKernel("mandel");

            messageSize = message.Length;

            // allocate a memory buffer with the message
            messageBuffer = new ComputeBuffer<int>(context,
                ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, message);

            kernel.SetMemoryArgument(0, messageBuffer);
            kernel.SetValueArgument(1, N);
        }

        static void GenerateMandelBrot(float ymin, float xmin, float width, int[] message)
        { 
            kernel.SetValueArgument(2, ymin);
            kernel.SetValueArgument(3, xmin);
            kernel.SetValueArgument(4, width);
            kernel.SetValueArgument(5, MaxIter);

            //var watch = System.Diagnostics.Stopwatch.StartNew();

            // Execute kernel
            for (var i = 0; i < N / 32; ++i)
            {
                for (var j = 0; j < N / 32; ++j)
                {
                    queue.Execute(kernel, new long[] { i * 32, j * 32 }, new long[] { 32, 32 }, null, null);
                }
            }
            
            // Read data back
            unsafe
            {
                fixed (int* retPtr = message)
                {
                    queue.Read(messageBuffer,
                        false, 0,
                        messageSize,
                        new IntPtr(retPtr),
                        null);

                    queue.Finish();
                }
            }

            //watch.Stop();

            //context.Dispose();
            //kernel.Dispose();
            //program.Dispose();
            //messageBuffer.Dispose();
            //queue.Dispose();

            //watch.Stop();
            //Console.WriteLine($"{watch.ElapsedMilliseconds} ms");
        }
    }
}
