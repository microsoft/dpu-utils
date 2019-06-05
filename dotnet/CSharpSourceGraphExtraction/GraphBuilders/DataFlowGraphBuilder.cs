using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using MSRC.DPU.Utils;

namespace MSRC.DPU.CSharpSourceGraphExtraction.GraphBuilders
{
    internal static class DataFlowGraphBuilder
    {
        public static void AddDataFlowEdges(SourceGraph sourceGraph, SyntaxNodeOrToken tokenOfInterest,
            ICollection<SyntaxNodeOrToken> forbiddenNodes = null,
            ICollection<Edge<SyntaxNodeOrToken, SourceGraphEdge>> addedEdges = null)
        {
            var semanticModel = sourceGraph.SemanticModel;

            //There's nothing before the declaration, so we don't need to bother:
            if (sourceGraph.VariableDeclarationNodes.Contains(tokenOfInterest))
            {
                return;
            }

            //We only ever need to visit each node once, so collect visited nodes here:
            var visitedNodes = new HashSet<(SyntaxNodeOrToken, bool)>();

            //Start from all predecessors of the token of interest:
            var toVisit = new Stack<(SyntaxNodeOrToken node, bool haveFoundUse)>();
            foreach (var (_, label, target) in sourceGraph.GetOutEdges(tokenOfInterest))
            {
                if (label != SourceGraphEdge.LastUsedVariable || (forbiddenNodes?.Contains(target) ?? false))
                {
                    continue;
                }
                if (visitedNodes.Add((target, false)))
                {
                    toVisit.Push((target, false));
                }
            }

            var nodeOfInterest = tokenOfInterest.IsToken ? tokenOfInterest.AsToken().Parent : tokenOfInterest.AsNode();
            ISymbol symbolToLookFor = nodeOfInterest != null ? semanticModel.GetSymbolInfo(nodeOfInterest).Symbol?.OriginalDefinition : null;
            string nodeLabelToLookFor = tokenOfInterest.ToString();

            while (toVisit.Count > 0)
            {
                var (node, haveFoundUse) = toVisit.Pop();
                var nodeSyntaxNode = node.IsToken ? node.AsToken().Parent : node.AsNode();
                var nodeSymbol = nodeSyntaxNode != null ? semanticModel.GetSymbolInfo(nodeSyntaxNode).Symbol?.OriginalDefinition : null;

                bool matches;
                if (symbolToLookFor == null || nodeSymbol == null)
                {
                    // This may happen in cases where Roslyn doesn't have symbol info
                    // or when one of the nodes is a dummy node (and thus doesn't belong to the SyntaxTree)
                    matches = node.ToString().Equals(nodeLabelToLookFor);
                }
                else
                {
                    matches = nodeSymbol.Equals(symbolToLookFor);
                }

                if (matches)
                {
                    if (!haveFoundUse)
                    {
                        var lastUseEdge = new Edge<SyntaxNodeOrToken, SourceGraphEdge>(tokenOfInterest, SourceGraphEdge.LastUse, node);
                        if (sourceGraph.AddEdge(lastUseEdge))
                        {
                            addedEdges?.Add(lastUseEdge);
                        }
                        haveFoundUse = true;
                    }

                    if (sourceGraph.VariableWriteNodes.Contains(node))
                    {
                        var lastWriteEdge = new Edge<SyntaxNodeOrToken, SourceGraphEdge>(tokenOfInterest, SourceGraphEdge.LastWrite, node);
                        if (sourceGraph.AddEdge(lastWriteEdge))
                        {
                            addedEdges?.Add(lastWriteEdge);
                        }
                        //We are done with this path -- we found a use and a write!
                        continue;
                    }
                    
                    //There's nothing before the declaration, so we don't need to bother to recurse further:
                    if (sourceGraph.VariableDeclarationNodes.Contains(node))
                    {
                        continue;
                    }
                }

                foreach (var (_, label, target) in sourceGraph.GetOutEdges(node))
                {
                    if (label != SourceGraphEdge.LastUsedVariable || (forbiddenNodes?.Contains(target) ?? false))
                    {
                        continue;
                    }
                    if (visitedNodes.Add((target, haveFoundUse))) {
                        toVisit.Push((target, haveFoundUse));
                    }
                }
            }
        }

        /// <summary>
        /// Adds LastUse/LastWrite dataflow edges to SourceGraph.
        /// Requires LastUsedVariables in the graph.
        /// </summary>
        public static void AddDataFlowGraph(SourceGraph sourceGraph)
        {
            foreach (var tokenOfInterest in sourceGraph.VariableUseNodes)
            {
                AddDataFlowEdges(sourceGraph, tokenOfInterest);
            }
        }
    }
}
