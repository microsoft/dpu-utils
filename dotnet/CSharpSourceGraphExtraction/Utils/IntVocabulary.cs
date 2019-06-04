using System;
using MSRC.DPU.Utils;

namespace MSRC.DPU.CSharpSourceGraphExtraction.Utils
{
    public class IntVocabulary<T> where T : class
    {
        private readonly BidirectionalMap<int, T> _dictionary = new BidirectionalMap<int, T>();
        private int _nextId = 0;

        public int Count => _dictionary.Count;

        public int Get(T obj, bool addIfNotPresent=false)
        {
            if (!_dictionary.TryGetKey(obj, out int key))
            {
                if (!addIfNotPresent)
                {
                    throw new Exception("Object not in vocabulary");
                }
                key = _nextId;
                _dictionary.Add(key, obj);
                _nextId++;
            }
            return key;
        }

        public bool Contains(T obj)
        {
            return _dictionary.Contains(obj);
        }

        public T Get(int objId)
        {
            return _dictionary.GetValue(objId);
        }
    }
}
