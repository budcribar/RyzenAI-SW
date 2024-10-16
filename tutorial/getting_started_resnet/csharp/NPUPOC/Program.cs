
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ResNetCIFAR
{
    class Program
    {
        // Constants related to CIFAR-10 images
        const int CIFAR_IMAGE_DEPTH = 3;
        const int CIFAR_IMAGE_WIDTH = 32;
        const int CIFAR_IMAGE_HEIGHT = 32;
        const int CIFAR_IMAGE_AREA = CIFAR_IMAGE_WIDTH * CIFAR_IMAGE_HEIGHT;
        const int CIFAR_LABEL_SIZE = 1;
        const int CIFAR_IMAGE_SIZE = CIFAR_IMAGE_DEPTH * CIFAR_IMAGE_AREA; // 3072 = 3 * 32 * 32

        // Lookup table for CIFAR-10 labels
        static readonly string[] CIFAR_LABELS = new string[]
        {
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        };

        static void Main(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("Usage: ResNetCIFAR <onnx model> <execution provider> <json_config> <img_url> [img_url]...");
                return;
            }

            // Set environment variables
            Environment.SetEnvironmentVariable("CONDA_PREFIX", "C:/Users/budcr/anaconda3/envs/ryzen-ai-1.2.0");
            string envVal = Environment.GetEnvironmentVariable("CONDA_PREFIX");
            Environment.SetEnvironmentVariable("PYTHONHOME", envVal);

            // Define paths
            string dataDir = "./data/cifar-10-batches-bin/test_batch.bin";
            string outputFolder = "images/";

            // Create output directory if it doesn't exist
            Directory.CreateDirectory(outputFolder);

            // Read first ten CIFAR-10 images
            var labeledImages = ReadFirstTenCIFAR10Images(dataDir);
            for (int i = 0; i < labeledImages.Count; i++)
            {
                string outputPath = Path.Combine(outputFolder, $"cifar_image_{i}.png");
                labeledImages[i].Image.Save(outputPath);
            }

            // Parse command-line arguments
            string modelPath = args[0];
            string ep = args[1].ToLower();
            string jsonConfig = args[2];

            if (!IsValidEP(ep))
            {
                Console.Error.WriteLine("Error: Choose from one of the available EP options: cpu, npu.");
                return;
            }

            Console.WriteLine($"Model path: {modelPath}");
            Console.WriteLine($"Execution Provider (EP): {ep}");

            // Initialize ONNX Runtime session options
            var sessionOptions = new SessionOptions();
         

            string cacheDir = Directory.GetCurrentDirectory();

            if (ep == "npu")
            {
                var options = new Dictionary<string, string>
                {
                    { "config_file", jsonConfig },
                    { "cacheDir", cacheDir },
                    { "cacheKey", "modelcachekey" }
                };
                try
                {
                    sessionOptions.AppendExecutionProvider_VitisAI(options);
                }
                catch (Exception e)
                {
                    Console.Error.WriteLine($"Exception occurred in appending execution provider: {e.Message}");
                }
            }

            // Create ONNX Runtime session
            using var session = new InferenceSession(modelPath, sessionOptions);

            // Get input and output information
            int inputCount = session.InputMetadata.Count;
            int outputCount = session.OutputMetadata.Count;

            var inputNames = session.InputMetadata.Keys.ToList();
            var outputNames = session.OutputMetadata.Keys.ToList();

            Console.WriteLine($"Input Nodes ({inputCount}):");
            foreach (var inputName in inputNames)
            {
                var shape = session.InputMetadata[inputName].Dimensions;
                Console.WriteLine($"\t{inputName} : {PrintShape(shape)}");
            }

            Console.WriteLine($"Output Nodes ({outputCount}):");
            foreach (var outputName in outputNames)
            {
                var shape = session.OutputMetadata[outputName].Dimensions;
                Console.WriteLine($"\t{outputName} : {PrintShape(shape)}");
            }

            // Process each image
            var results = new List<(string PredictedLabel, string ActualLabel)>();

            for (int i = 0; i < labeledImages.Count; i++)
            {
                string currentFile = Path.Combine(outputFolder, $"cifar_image_{i}.png");
                var inputShape = session.InputMetadata[inputNames[0]].Dimensions.ToList();

                if (inputShape[0] == -1)
                {
                    inputShape[0] = 1; // Batch size
                }

                int totalElements = CalculateProduct(inputShape);
                var inputTensorValues = new float[totalElements];
                PreprocessResNet(currentFile, inputTensorValues, inputShape);

                // Create tensor from input data
                var tensor = new DenseTensor<float>(inputTensorValues, inputShape.ToArray());

                // Run inference
                try
                {
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor(inputNames[0], tensor)
                    };

                    var outputs = session.Run(inputs);

                    // Assume single output
                    var outputTensor = outputs.First().AsEnumerable<float>().ToArray();
                    var softmaxOutput = Softmax(outputTensor);
                    var top1 = TopK(softmaxOutput, 1).First();

                    string predicted = Lookup(top1.Index);
                    string actual = Lookup(labeledImages[i].Label);

                    results.Add((predicted, actual));

                    outputs.Dispose();
                }
                catch (OnnxRuntimeException e)
                {
                    Console.Error.WriteLine($"ERROR running model inference: {e.Message}");
                    return;
                }
            }

            // Display results
            Console.WriteLine("Final Results:");
            for (int i = 0; i < results.Count; i++)
            {
                Console.WriteLine($"Image {i}: Predicted Label = {results[i].PredictedLabel}, Actual Label = {results[i].ActualLabel}");
            }

            // Reset environment variable
            Environment.SetEnvironmentVariable("PYTHONHOME", "");
        }

        /// <summary>
        /// Validates the execution provider.
        /// </summary>
        static bool IsValidEP(string option)
        {
            var validOptions = new List<string> { "cpu", "npu" };
            return validOptions.Contains(option);
        }

        /// <summary>
        /// Reads the first ten CIFAR-10 images from the binary file.
        /// </summary>
        static List<(Image<Bgr, byte> Image, int Label)> ReadFirstTenCIFAR10Images(string filename)
        {
            var labeledImages = new List<(Image<Bgr, byte>, int)>();

            try
            {
                using var file = new FileStream(filename, FileMode.Open, FileAccess.Read);
                using var reader = new BinaryReader(file);

                for (int count = 0; count < 10; count++)
                {
                    if (reader.BaseStream.Position + CIFAR_LABEL_SIZE + CIFAR_IMAGE_SIZE > reader.BaseStream.Length)
                    {
                        Console.Error.WriteLine("Reached end of file or insufficient data.");
                        break;
                    }

                    byte label = reader.ReadByte();
                    byte[] data = reader.ReadBytes(CIFAR_IMAGE_SIZE);

                    // Split data into channels
                    byte[] r = new byte[CIFAR_IMAGE_AREA];
                    byte[] g = new byte[CIFAR_IMAGE_AREA];
                    byte[] b = new byte[CIFAR_IMAGE_AREA];

                    Buffer.BlockCopy(data, 0, r, 0, CIFAR_IMAGE_AREA);
                    Buffer.BlockCopy(data, CIFAR_IMAGE_AREA, g, 0, CIFAR_IMAGE_AREA);
                    Buffer.BlockCopy(data, 2 * CIFAR_IMAGE_AREA, b, 0, CIFAR_IMAGE_AREA);

                    // Create BGR image
                    var img = new Image<Bgr, byte>(CIFAR_IMAGE_WIDTH, CIFAR_IMAGE_HEIGHT);
                    img.Data = new byte[CIFAR_IMAGE_HEIGHT, CIFAR_IMAGE_WIDTH, 3];

                    for (int h = 0; h < CIFAR_IMAGE_HEIGHT; h++)
                    {
                        for (int w = 0; w < CIFAR_IMAGE_WIDTH; w++)
                        {
                            img.Data[h, w, 0] = b[h * CIFAR_IMAGE_WIDTH + w]; // B
                            img.Data[h, w, 1] = g[h * CIFAR_IMAGE_WIDTH + w]; // G
                            img.Data[h, w, 2] = r[h * CIFAR_IMAGE_WIDTH + w]; // R
                        }
                    }

                    labeledImages.Add((img, label));
                }
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"Error reading CIFAR-10 data: {e.Message}");
            }

            return labeledImages;
        }

        /// <summary>
        /// Preprocesses the image for ResNet.
        /// </summary>
        static void PreprocessResNet(string file, float[] inputTensorValues, List<int> inputShape)
        {
            using var image = ReadImage(file);
            var processedImage = PreprocessImage(image, new System.Drawing.Size(inputShape[3], inputShape[2]));
            SetInputImage(processedImage, inputTensorValues);
        }

        /// <summary>
        /// Postprocesses the model output to obtain the predicted label.
        /// </summary>
        static string PostprocessResNet(float[] outputTensor)
        {
            var softmaxOutput = Softmax(outputTensor);
            var top1 = TopK(softmaxOutput, 1).First();
            return Lookup(top1.Index);
        }

        /// <summary>
        /// Reads an image from the file path.
        /// </summary>
        static Image<Bgr, byte> ReadImage(string file)
        {
            return new Image<Bgr, byte>(file);
        }

        /// <summary>
        /// Crops the image to the specified height and width.
        /// </summary>
        static Image<Bgr, byte> CropImage(Image<Bgr, byte> image, int height, int width)
        {
            int offsetH = (image.Height - height) / 2;
            int offsetW = (image.Width - width) / 2;
            var rect = new System.Drawing.Rectangle(offsetW, offsetH, width, height);
            return image.Copy(rect);
        }

        /// <summary>
        /// Preprocesses the image by resizing and cropping.
        /// </summary>
        static Image<Bgr, byte> PreprocessImage(Image<Bgr, byte> image, System.Drawing.Size size)
        {
            float smallestSide = 256f;
            float scale = smallestSide / Math.Min(image.Width, image.Height);
            int newWidth = (int)(image.Width * scale);
            int newHeight = (int)(image.Height * scale);
            var resizedImage = image.Resize(newWidth, newHeight, Inter.Linear);
            return CropImage(resizedImage, size.Height, size.Width);
        }

        /// <summary>
        /// Sets the input image data into the tensor.
        /// </summary>
        static void SetInputImage(Image<Bgr, byte> image, float[] data)
        {
            float[] mean = { 0.0f, 0.0f, 0.0f };
            float[] scales = { 1.0f, 1.0f, 1.0f };

            int width = image.Width;
            int height = image.Height;

            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int index = c * height * width + h * width + w;
                        int channel = 2 - c; // BGR to RGB
                        float pixel = (image.Data[h, w, channel] - mean[channel]) * scales[channel] / 255f;
                        data[index] = pixel;
                    }
                }
            }
        }

        /// <summary>
        /// Applies softmax to the input data.
        /// </summary>
        static float[] Softmax(float[] data)
        {
            float max = data.Max();
            var exps = data.Select(d => (float)Math.Exp(d - max)).ToArray();
            float sum = exps.Sum();
            return exps.Select(e => e / sum).ToArray();
        }

        /// <summary>
        /// Retrieves the top K indices and their corresponding scores.
        /// </summary>
        static List<(int Index, float Score)> TopK(float[] scores, int K)
        {
            return scores
                .Select((score, index) => (Index: index, Score: score))
                .OrderByDescending(x => x.Score)
                .Take(K)
                .ToList();
        }

        /// <summary>
        /// Looks up the label based on the index.
        /// </summary>
        static string Lookup(int index)
        {
            if (index < 0 || index >= CIFAR_LABELS.Length)
                return string.Empty;
            return CIFAR_LABELS[index];
        }

        /// <summary>
        /// Prints the shape of the tensor.
        /// </summary>
        static string PrintShape(int[] shape)
        {
            return string.Join("x", shape);
        }

        /// <summary>
        /// Calculates the product of elements in the shape list.
        /// </summary>
        static int CalculateProduct(List<int> shape)
        {
            return shape.Aggregate(1, (acc, val) => acc * val);
        }
    }

    /// <summary>
    /// Extension methods for SessionOptions to append VitisAI Execution Provider.
    /// </summary>
    public static class SessionOptionsExtensions
    {
        /// <summary>
        /// Appends the VitisAI Execution Provider with the given options.
        /// </summary>
        public static void AppendExecutionProvider_VitisAI(this SessionOptions options, Dictionary<string, string> epOptions)
        {
            // Note: VitisAI Execution Provider is hypothetical in this context.
            // Replace with actual implementation based on the ONNX Runtime's VitisAI provider.
            throw new NotImplementedException("VitisAI Execution Provider integration is not implemented.");
        }
    }
}
