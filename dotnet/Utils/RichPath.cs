using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using Newtonsoft.Json;
using Microsoft.Azure.Storage.Auth;
using Microsoft.Azure.Storage;
using Microsoft.Azure.Storage.Blob;
using System.IO.Compression;
using System.Linq;

namespace MSRC.DPU.Utils
{
    public abstract class RichPath
    {
        public const string AZURE_PATH_PREFIX = "azure://";

        public static RichPath Create(string path, string azureInfoPath = null)
        {
            if (path.StartsWith(AZURE_PATH_PREFIX))
            {
                if (azureInfoPath == null)
                {
                    throw new ArgumentException($"Azure path {path} requires azure authentification info.");
                }
                return new AzurePath(azureInfoPath, path);
            }
            else
            {
                return new LocalPath(path);
            }
        }

        public abstract string Path { get; }

        public abstract bool IsDirectory();

        public abstract bool IsFile();

        public abstract void MakeAsDirectory();

        public abstract Stream ReadAsBinaryPotentiallyUncompress();

        public abstract void Store(object data);

        public abstract IEnumerable<RichPath> FilesInDirectory(string searchPattern = "*");

        public abstract RichPath Join(string newComponent);

        protected static void StoreAsJSON(Stream outStream, object data, bool jsonl = false)
        {
            var serializer = new JsonSerializer { Formatting = Formatting.None }; // No whitespace, no linebreaks
            using (var textWriter = new StreamWriter(outStream))
            {
                if (!jsonl)
                {
                    serializer.Serialize(textWriter, data);
                }
                else
                {
                    if (data is System.Collections.ICollection collection)
                    {
                        foreach (var ele in collection)
                        {
                            serializer.Serialize(textWriter, ele);
                            textWriter.WriteLine();
                        }
                    }
                    else
                    {
                        throw new ArgumentException($"Only data implementing the ICollection interface can be serialized as JSONL. Provided data has type {data.GetType()}.");
                    }
                }
            }
        }
        protected static T ReadJSONFromTextReader<T>(TextReader reader)
        {
            using (var jsonReader = new JsonTextReader(reader))
            {
                var deserializer = new JsonSerializer();
                return deserializer.Deserialize<T>(jsonReader);
            }
        }

        protected static IEnumerable<T> ReadJSONLFromTextReader<T>(TextReader reader)
        {
            string dataLine;
            while ((dataLine = reader.ReadLine()) != null)
            {
                yield return JsonConvert.DeserializeObject<T>(dataLine);
            }
        }

        public T Read<T>()
        {
            using (var inStream = ReadAsBinaryPotentiallyUncompress())
            using (var textReader = new StreamReader(inStream))
            {
                return ReadJSONFromTextReader<T>(textReader);
            }
        }

        public IEnumerable<T> ReadCollection<T>()
        {
            using (var inStream = ReadAsBinaryPotentiallyUncompress())
            using (var textReader = new StreamReader(inStream))
            {
                if (Path.EndsWith(".jsonl") || Path.EndsWith(".jsonl.gz"))
                {
                    // We need to loop over these explicitly so that used textReader doesn't get disposed.
                    foreach (var element in ReadJSONLFromTextReader<T>(textReader))
                    {
                        yield return element;
                    }
                }
                else if (Path.EndsWith(".json") || Path.EndsWith(".json.gz"))
                {
                    foreach (var element in ReadJSONFromTextReader<List<T>>(textReader))
                    {
                        yield return element;
                    }
                }
                else
                {
                    throw new ArgumentException($"File suffix must be .json[.gz] or .jsonl[.gz], but got {Path}");
                }
            }
        }

        protected static bool StreamIsGZipped(Stream inStream)
        {
            // Read first two bytes as signature, and seek to beginning again:
            byte[] signature = new byte[2];
            inStream.Read(signature, 0, 2);
            inStream.Seek(0, SeekOrigin.Begin);
            return signature[0] == 0x1f && signature[1] == 0x8b;
        }
    }

    public class LocalPath : RichPath
    {
        public override string Path { get; }

        public LocalPath(string path)
        {
            Path = path;
        }

        public override string ToString() => Path;

        public override bool IsDirectory() => Directory.Exists(Path);

        public override bool IsFile() => File.Exists(Path);

        public override void MakeAsDirectory() => Directory.CreateDirectory(Path);

        public override Stream ReadAsBinaryPotentiallyUncompress()
        {
            var inStream = File.OpenRead(Path);

            if (StreamIsGZipped(inStream))
            {
                return new GZipStream(inStream, CompressionMode.Decompress);
            }

            return inStream;
        }

        public override void Store(object data)
        {
            using (var outStream = new FileStream(Path, FileMode.Create))
            {
                string normalizedPath;
                Stream possiblyCompressedStream;
                if (Path.EndsWith(".gz"))
                {
                    normalizedPath = Path.Substring(0, Path.Length - 3); // Remove the ".gz"
                    possiblyCompressedStream = new GZipStream(outStream, CompressionMode.Compress);
                }
                else
                {
                    normalizedPath = Path;
                    possiblyCompressedStream = outStream;
                }

                if (normalizedPath.EndsWith(".json"))
                {
                    StoreAsJSON(possiblyCompressedStream, data, jsonl: false);
                }
                else if (normalizedPath.EndsWith(".jsonl"))
                {
                    StoreAsJSON(possiblyCompressedStream, data, jsonl: true);
                }
                else
                {
                    throw new ArgumentException($"File suffix must be .json[.gz] or .jsonl[.gz], but got {Path}");
                }
            }
        }

        public override IEnumerable<RichPath> FilesInDirectory(string searchPattern = "*")
        {
            foreach (var newPath in Directory.EnumerateFiles(Path, searchPattern, SearchOption.AllDirectories))
            {
                yield return new LocalPath(newPath);
            }
        }

        public override RichPath Join(string newComponent) 
            => new LocalPath(System.IO.Path.Combine(Path, newComponent));
    }

    public class AzurePath : RichPath
    {
        public override string Path { get; }
        private readonly string accountName;
        private readonly string containerName;
        private readonly string cacheLocation;
        private readonly CloudBlobContainer blobContainerClient;

        public AzurePath(string azureInfoPath, string path)
        {
            if (!path.StartsWith(AZURE_PATH_PREFIX))
            {
                throw new ArgumentException($"Azure paths need to be in the format {AZURE_PATH_PREFIX}/account_name/container_name/path, but got {path}.");
            }

            // Strip off azure:// prefix:
            var accountPath = path.Substring(AZURE_PATH_PREFIX.Length);
            var pathParts = accountPath.Split(new char[] { '/' }, 3);

            if (pathParts.Length < 3)
            {
                throw new ArgumentException($"Azure paths need to be in the format {AZURE_PATH_PREFIX}/account_name/container_name/path, but got {path}.");
            }
            var (accountName, containerName, containerPath) = (pathParts[0], pathParts[1], pathParts[2]);

            using (var azureInfoInstream = File.OpenText(azureInfoPath))
            using (var azureInfoJsonReader = new JsonTextReader(azureInfoInstream))
            {
                var deserializer = new JsonSerializer();
                var azureInfo = deserializer.Deserialize<Dictionary<string, Dictionary<string, string>>>(
                    azureInfoJsonReader);

                if (!azureInfo.TryGetValue(accountName, out var accountInfo))
                {
                    throw new ArgumentException($"Could not find access information for account '{accountName}'!");
                }

                StorageCredentials storageCredentials;
                if (accountInfo.TryGetValue("sas_token", out var sasToken))
                {
                    storageCredentials = new StorageCredentials(sasToken);
                }
                else if (accountInfo.TryGetValue("account_key", out var accountKey))
                {
                    storageCredentials = new StorageCredentials(accountName: accountName, keyValue: accountKey);
                }
                else
                {
                    throw new ArgumentException(
                        $"Access to Azure storage account {accountName} requires either account_key or sas_token!");
                }

                var storageAccount = new CloudStorageAccount(storageCredentials, useHttps: true, accountName: accountName, endpointSuffix: "core.windows.net");

                if (accountInfo.TryGetValue("cache_location", out string cacheLocation))
                {
                    // Replace environment variables in the cache location
                    cacheLocation = Regex.Replace(
                        cacheLocation,
                        "${([^}]+)}",
                        match => Environment.GetEnvironmentVariable(match.Groups[0].Value) ?? match.Value);
                }

                this.accountName = accountName;
                this.containerName = containerName;
                this.cacheLocation = cacheLocation;
                blobContainerClient = storageAccount.CreateCloudBlobClient().GetContainerReference(this.containerName);
                Path = containerPath;
            }
        }

        private AzurePath(AzurePath oldAzurePath, string newPath)
        {
            accountName = oldAzurePath.accountName;
            containerName = oldAzurePath.containerName;
            cacheLocation = oldAzurePath.cacheLocation;
            blobContainerClient = oldAzurePath.blobContainerClient;
            Path = newPath;
        }

        public override string ToString()
            => $"{AZURE_PATH_PREFIX}{accountName}/{containerName}/{Path}";

        public override bool IsDirectory()
        {
            var listingResult = blobContainerClient.ListBlobs(Path).FirstOrDefault();
            if (listingResult == null)
            {
                return false;  // Doesn't exist, so we return false as LocalPath.IsDirectory()
            }
            else
            {
                if (listingResult is CloudBlobDirectory)
                {
                    return true;
                }
                else if ((listingResult as CloudBlob)?.Name.Equals(Path) ?? false)
                {
                    return false;  // The only result is the blob itself -> It's a file
                }
            }
            return true;
        }

        public override bool IsFile()
        {
            var listingResult = blobContainerClient.ListBlobs(Path).FirstOrDefault();
            if (listingResult == null)
            {
                return false;  // Doesn't exist
            }
            else
            {
                if (listingResult is CloudBlobDirectory)
                {
                    return false;
                }
                return true;
            }
        }

        public override void MakeAsDirectory() 
        {
            return;  // Nothing to do in BlobStorage
        }

        private Stream ReadDirectlyFromBlobStoragePotentiallyUncompress()
        {
            var blobReference = blobContainerClient.GetBlobReference(Path);
            var memoryStream = new MemoryStream();
            blobReference.DownloadToStream(memoryStream);
            memoryStream.Seek(0, SeekOrigin.Begin);

            if (StreamIsGZipped(memoryStream))
            {
                return new GZipStream(memoryStream, CompressionMode.Decompress);
            }

            return memoryStream;
        }

        private string CachedFilePath => 
            System.IO.Path.Combine(
                cacheLocation,
                containerName,
                Path.Replace('/', System.IO.Path.DirectorySeparatorChar) // Note that this translates a blob path (/ separators) to a local path (potentially using \)
                );
        private string CachedFileEtagPath => CachedFilePath + ".etag";

        private LocalPath CacheFileLocally()
        {
            string storedEtag = null;
            if (File.Exists(CachedFileEtagPath))
            {
                storedEtag = File.ReadAllText(CachedFileEtagPath);
            }

            var tmpPath = CachedFilePath + Guid.NewGuid();
            try
            {
                Directory.CreateDirectory(System.IO.Path.GetDirectoryName(CachedFilePath));
                // Download blob to a tmp file, to avoid overwriting a potentially existing copy:
                var blobReference = blobContainerClient.GetBlobReference(Path);
                blobReference.DownloadToFile(
                    tmpPath,
                    FileMode.Create,
                    new AccessCondition { IfNoneMatchETag = storedEtag });
                // If download succeeded (no exception), remove old file if it exists and put new file in place:
                if (File.Exists(CachedFilePath)) {
                    File.Delete(CachedFilePath);
                }
                File.Move(tmpPath, CachedFilePath);
                File.WriteAllText(CachedFileEtagPath, blobReference.Properties.ETag);
            }
            catch (StorageException e)
            {
                if (e.RequestInformation.HttpStatusCode != 304) // 304 ~ Not modified, i.e., we are fine
                {
                    throw;
                }
            }
            finally
            { 
                if (File.Exists(tmpPath))
                {
                    File.Delete(tmpPath);
                }
            }

            return new LocalPath(CachedFilePath);
        }

        public override Stream ReadAsBinaryPotentiallyUncompress()
        {
            if (cacheLocation == null)
            {
                return ReadDirectlyFromBlobStoragePotentiallyUncompress();
            }
            return CacheFileLocally().ReadAsBinaryPotentiallyUncompress();
        }

        public override void Store(object data)
        {
            // Create new file that has same extension as this file...
            var tempFile = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                Guid.NewGuid().ToString() + Path.Split('/').Last());
            try
            {
                var tempLocalPath = new LocalPath(tempFile);
                tempLocalPath.Store(data);
                var blobReference = blobContainerClient.GetBlockBlobReference(Path);
                blobReference.UploadFromFile(tempFile);

                // Also store to local cache, if we have one:
                if (cacheLocation != null)
                {
                    Directory.CreateDirectory(System.IO.Path.GetDirectoryName(CachedFilePath));
                    if (File.Exists(CachedFilePath))
                    {
                        File.Delete(CachedFilePath);
                    }
                    File.Move(tempFile, CachedFilePath);
                    File.WriteAllText(CachedFileEtagPath, blobReference.Properties.ETag);
                }
            }
            finally
            {
                if (File.Exists(tempFile))
                {
                    File.Delete(tempFile);
                }
            }
        }

        public override IEnumerable<RichPath> FilesInDirectory(string searchPattern = "*")
        {
            string fullPattern = Path + "/" + searchPattern;
            var glob = DotNet.Globbing.Glob.Parse(fullPattern);
            foreach (var blob in blobContainerClient.ListBlobs(Path, useFlatBlobListing: true))
            {

                if (blob is CloudBlob cloudBlob && glob.IsMatch(cloudBlob.Name))
                {
                    yield return new AzurePath(this, cloudBlob.Name);
                }
            }
        }

        public override RichPath Join(string newComponent) 
            => new AzurePath(this, Path + "/" + newComponent);
    }
}
