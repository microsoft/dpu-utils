using System;
using System.Collections.Generic;

namespace MSRC.DPU.Utils
{
    public static class ExtensionUtils
    {
        public static void Deconstruct<K, V>(this KeyValuePair<K, V> kvp, out K key, out V value)
        {
            key = kvp.Key;
            value = kvp.Value;
        }

        public static V TryGetOrAddValue<K, V>(this Dictionary<K, V> dict, K key, out V value, Func<V> computeDefault)
        {
            if (!dict.TryGetValue(key, out value))
            {
                value = computeDefault();
                dict.Add(key, value);                
            }
            return value;
        }
    }
}
