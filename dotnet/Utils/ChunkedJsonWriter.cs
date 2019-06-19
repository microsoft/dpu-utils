using Newtonsoft.Json;
using System;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace MSRC.DPU.Utils
{
    /// <summary>
    /// Thread-safe .json[l].gz writer. The output is automatically split in chunks.
    /// </summary>
    public class ChunkedJsonGzWriter : IDisposable
    {
        private readonly object _lock = new object();
        private TextWriter _textStream = null;
        private int _numElementsWrittenInCurrentChunk = 0;
        private readonly string _outputFilenameTemplate;

        private readonly int _max_elements_per_chunk;
        private readonly bool _useJsonlFormat;

        public ChunkedJsonGzWriter(string outputFilenameTemplate,
            int max_elements_per_chunk = 500,
            bool useJsonlFormat = true,
            bool resumeIfFilesExist = false)
        {
            _outputFilenameTemplate = outputFilenameTemplate;
            _max_elements_per_chunk = max_elements_per_chunk;
            _useJsonlFormat = useJsonlFormat;
            if (resumeIfFilesExist)
            {
                // Loop Until there is an unwritten file
                for (int i = 0; ; i++)
                {
                    if (File.Exists(GetChunkedOutputFilename(_outputFilenameTemplate, NumChunksWrittenSoFar)))
                    {
                        NumChunksWrittenSoFar++;
                    }
                    else
                    {
                        break;
                    }
                }
            }

        }

        public int NumChunksWrittenSoFar { get; private set; } = 0;

        /// <summary>
        /// Write JSON representation of a single datapoint to the output. The method handles details of chunking.
        /// </summary>
        /// <param name="jsonElement">String containing a JSON-encoded data point.</param>
        public void WriteElement(string jsonElement)
        {
            lock (_lock)
            {
                if (_textStream == null)
                {
                    var filename = GetChunkedOutputFilename(_outputFilenameTemplate, NumChunksWrittenSoFar);
                    Console.WriteLine($"Opening output file {filename}.");
                    var fileStream = File.Create(filename);
                    var gzipStream = new GZipStream(fileStream, CompressionMode.Compress, false);
                    _textStream = new StreamWriter(gzipStream);
                    _numElementsWrittenInCurrentChunk = 0;
                    if (!_useJsonlFormat) _textStream.Write('[');
                }

                if (_numElementsWrittenInCurrentChunk > 0)
                {
                    if (_useJsonlFormat)
                    {
                        _textStream.Write('\n');
                    }
                    else
                    {
                        _textStream.Write(',');
                    }
                }
                _textStream.Write(jsonElement);

                ++_numElementsWrittenInCurrentChunk;
                if (_numElementsWrittenInCurrentChunk >= _max_elements_per_chunk)
                {
                    CloseOutputFile();
                }
            }
        }

        /// <summary>
        /// Write JSON representation of a single datapoint to the output. The method handles details of chunking.
        /// </summary>
        /// <param name="writer">A callback that writes some data to the provided JsonWriter, for example your hand-rolled serialization code.</param>
        public void WriteElement(Action<JsonWriter> writer)
        {
            string jsonElement;
            using (MemoryStream ms = new MemoryStream())
            {
                TextWriter tw = new StreamWriter(ms);
                JsonWriter js = new JsonTextWriter(tw);

                writer(js);
                js.Flush();
                ms.Seek(0, SeekOrigin.Begin);

                using (TextReader sr = new StreamReader(ms))
                {
                    jsonElement = sr.ReadToEnd();
                }

                WriteElement(jsonElement);
            }
        }

        private string GetChunkedOutputFilename(string fileName, int chunkNum)
        {
            var outputFormat = (_useJsonlFormat ? ".jsonl" : ".json") + ".gz";
            if (fileName.EndsWith(outputFormat))
            {
                return fileName.Replace(outputFormat, "." + chunkNum + outputFormat);
            }
            else
            {
                return fileName + "." + chunkNum + outputFormat;
            }
        }

        private void CloseOutputFile()
        {
            lock (_lock)
            {
                if (!_useJsonlFormat) _textStream.Write(']');
                _textStream.Close();
                _textStream = null;
                ++NumChunksWrittenSoFar;
            }
        }

        public void Dispose()
        {
            if (_textStream != null)
            {
                CloseOutputFile();
            }
        }
    }
}
