using Cloo;
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Input;
using System.Diagnostics;

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
        static int maxIter = 32*256;
        static int N = 1024;

        static WriteableBitmap writeableBitmap;
        static Window window;
        static Image image;
        static Label debugLabel;
        static Panel grid;

        static double ymin;
        static double xmin;
        static double width; 
        static int[] message;
        static int[] gradient;

        static int gradientLength;

        static double mouseymin;
        static double mousexmin;
        static Point mouseOrig;
        static bool moving;

        static ComputePlatform platform;
        static ComputeCommandQueue queue;
        static ComputeContext context;
        static ComputeProgram program;
        static ComputeKernel kernel;
        static ComputeBuffer<int> messageBuffer;
        static ComputeBuffer<int> gradientBuffer;
        static int messageSize;

        static bool useDoublePrecision = false;
        static int coclorCycle = 0;
        static int colorCycleFrameIncrement = 0;

        [STAThread]
        static void Main(string[] args)
        {
            grid = new Grid();

            image = new Image();
            RenderOptions.SetBitmapScalingMode(image, BitmapScalingMode.NearestNeighbor);
            RenderOptions.SetEdgeMode(image, EdgeMode.Aliased);

            debugLabel = new Label();
            debugLabel.Content = "Debug information";
            debugLabel.Foreground = new SolidColorBrush(Colors.White);
            debugLabel.Background = new SolidColorBrush(Color.FromArgb(150,100,100,100));
            debugLabel.Width = 240;
            debugLabel.Height = 28;
            debugLabel.VerticalAlignment = VerticalAlignment.Top;

            grid.Children.Add(image);
            grid.Children.Add(debugLabel);

            window = new Window()
            {
                Width = N,
                Height = N,
                Title = "MandelCuda - Mandelbrot rendered with nvidia CUDA",
                ResizeMode = ResizeMode.NoResize
            };
            window.Content = grid;
            window.Show();

            writeableBitmap = new WriteableBitmap(
                (int)window.ActualWidth,
                (int)window.ActualHeight,
                96,
                96,
                PixelFormats.Bgr32,
                null);

            image.Source = writeableBitmap;

            image.Stretch = Stretch.None;
            image.HorizontalAlignment = HorizontalAlignment.Left;
            image.VerticalAlignment = VerticalAlignment.Top;

            image.MouseMove += I_MouseMove;
            image.MouseLeftButtonDown += (_, e) => 
            { 
                mouseOrig = e.GetPosition(window); 
                mouseymin = ymin; 
                mousexmin = xmin; 
                moving = true; 
            };
            image.MouseLeftButtonUp += (_, __) => moving = false;

            CreateColorGradient(new Color[] { Colors.Blue, Colors.White, Colors.Red }, 32);

            window.MouseWheel += W_MouseWheel;

            ymin = -2f;
            xmin = -2f;
            width = 4f;
            message = new int[N * N];
            messageSize = message.Length;

            SetupCUDA(useDoublePrecision ? "Mandel4d.cl" : "Mandel4s.cl");
            kernel.SetMemoryArgument(0, messageBuffer);
            kernel.SetValueArgument(1, N);
            kernel.SetMemoryArgument(2, gradientBuffer);
            kernel.SetValueArgument(3, gradientLength);                    

            GenerateMandelBrot();
            
            Application app = new Application();
            app.Run();
            
            TearDownCUDA();
        }

        private static void I_MouseMove(object sender, MouseEventArgs e)
        {
            if (moving)
            {
                var pos = e.GetPosition(window);
                xmin = mousexmin - (float)(pos.X - mouseOrig.X) * (width / N);
                ymin = mouseymin - (float)(pos.Y - mouseOrig.Y) * (width / N);

                GenerateMandelBrot();
            }
        }

        private static void W_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            // e.Delta is a multiplum of 120
            if (e.Delta == 0)
            {
                return;
            }

            var pos = e.GetPosition(window);
            const float minm = 0.9f;
            float m = (float)Math.Pow(minm, e.Delta / 120);

            float y = (float)(pos.Y / N);
            float x = (float)(pos.X / N);

            double ycenter = ymin + width * y;
            double xcenter = xmin + width * x;

            width = width * m;
            ymin = ycenter - width*y;
            xmin = xcenter - width*x;

            GenerateMandelBrot();
        }

        private static void CreateColorGradient(Color[] colors, int valuesPerColor)
        {
            gradientLength = colors.Length * valuesPerColor;
            gradient = new int[gradientLength];
            for (int c = 0; c < colors.Length; ++c)
            {
                for (int v = 0; v < valuesPerColor; ++v)
                {
                    var c0 = colors[c];
                    var c1 = colors[(c+1) % colors.Length];

                    var d0 = (valuesPerColor-v) / (float)valuesPerColor;
                    var d1 = v / (float)valuesPerColor;
                    gradient[c*valuesPerColor + v] = ((int)(c0.R*d0+c1.R*d1)) << 16
                        | ((int)(c0.G*d0 + c1.G * d1)) << 8
                        | (int)(c0.B * d0 + c1.B*d1);

                }
            }
        }

        static void TearDownCUDA()
        {
            context.Dispose();
            queue.Dispose();
            kernel.Dispose();
            messageBuffer.Dispose();
            if (gradientBuffer != null)
            {
                gradientBuffer.Dispose();
            }
            program.Dispose();
        }

        // 26 ms 4096x4096@512 iter with 1024 cores
        static void SetupCUDA(string sourceFile)
        {
            //var watch = System.Diagnostics.Stopwatch.StartNew();
            // pick first platform
            platform = ComputePlatform.Platforms[0];

            // create context with all gpu devices
            context = new ComputeContext(ComputeDeviceTypes.Gpu,
                new ComputeContextPropertyList(platform), null, IntPtr.Zero);   // LEAK

            // create a command queue with first gpu found
            queue = new ComputeCommandQueue(context,
            context.Devices[0], ComputeCommandQueueFlags.None);

            // load opencl source
            using (var streamReader = new StreamReader(sourceFile))
            {
                string clSource = streamReader.ReadToEnd();

                // create program with opencl source
                program = new ComputeProgram(context, clSource);

                // compile opencl source
                program.Build(null, null, null, IntPtr.Zero);

                // load chosen kernel from program
                kernel = program.CreateKernel("mandel");

                // allocate a memory buffer with the message
                messageBuffer = new ComputeBuffer<int>(context,
                    ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.UseHostPointer, message);
                gradientBuffer = new ComputeBuffer<int>(context,
                    ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, gradient);
                streamReader.Close();
            }
        }

        static void GenerateMandelBrot()
        {
            var nWidth = width / N;

            if (useDoublePrecision)
            {
                kernel.SetValueArgument(4, ymin);
                kernel.SetValueArgument(5, xmin);
                kernel.SetValueArgument(6, nWidth);
            }
            else
            {
                kernel.SetValueArgument(4, (float)ymin);
                kernel.SetValueArgument(5, (float)xmin);
                kernel.SetValueArgument(6, (float)nWidth);
            }
            kernel.SetValueArgument(7, maxIter);
            kernel.SetValueArgument(8, coclorCycle);

            coclorCycle += colorCycleFrameIncrement;

            Stopwatch watch = new Stopwatch();

            watch.Start();

            // Execute kernel
            queue.Execute(kernel, new long[] { 0, 0 }, new long[] { N, N }, null, null);

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

            watch.Stop();
            debugLabel.Content = $"{watch.ElapsedMilliseconds} ms";

            // Write to bitmap
            try
            {
                // Reserve the back buffer for updates.
                writeableBitmap.Lock();

                unsafe
                {
                    IntPtr buffer = writeableBitmap.BackBuffer;
                    System.Runtime.InteropServices.Marshal.Copy(message, 0, buffer, (int)(writeableBitmap.Height * writeableBitmap.Width));
                }

                // Specify the area of the bitmap that changed.
                writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, (int)writeableBitmap.Width, (int)writeableBitmap.Height));
            }
            finally
            {
                // Release the back buffer and make it available for display.
                writeableBitmap.Unlock();
            }
        }
    }
}
