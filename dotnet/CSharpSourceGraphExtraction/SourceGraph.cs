using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Collections.Concurrent;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Newtonsoft.Json;
using MSRC.DPU.Utils;
using MSRC.DPU.CSharpSourceGraphExtraction.Utils;
using MSRC.DPU.CSharpSourceGraphExtraction.GraphBuilders;

namespace MSRC.DPU.CSharpSourceGraphExtraction
{
    public enum SourceGraphEdge
    {
        Child,
        NextToken,
        LastUsedVariable,
        LastUse,
        LastWrite,
        LastLexicalUse,
        ComputedFrom,
        ReturnsTo,
        GuardedBy,
        GuardedByNegation
    }

    public readonly struct MethodInvocationInformation
    {
        public readonly IMethodSymbol InvokedMethodSymbol;
        public readonly SyntaxNode MethodInvocationNode;
        public readonly IDictionary<SyntaxNodeOrToken, (string parameterName, ITypeSymbol parameterType)> ArgumentNodeToMethodParameter;

        public MethodInvocationInformation(
            IMethodSymbol invokedMethodSymbol,
            SyntaxNode methodInvocationNode,
            IDictionary<SyntaxNodeOrToken, (string parameterName, ITypeSymbol parameterType)> argumentNodeToMethodParameter)
        {
            InvokedMethodSymbol = invokedMethodSymbol;
            MethodInvocationNode = methodInvocationNode;
            ArgumentNodeToMethodParameter = argumentNodeToMethodParameter;
        }

        public void ReplaceSyntaxNodes(IReadOnlyDictionary<SyntaxNode, SyntaxNodeOrToken> nodeToReplacement)
        {
            Debug.Assert(!nodeToReplacement.ContainsKey(MethodInvocationNode));
            foreach (var parameter in ArgumentNodeToMethodParameter.Where(kv => kv.Key.IsNode && nodeToReplacement.ContainsKey(kv.Key.AsNode())).ToArray())
            {
                ArgumentNodeToMethodParameter.Remove(parameter.Key);
                ArgumentNodeToMethodParameter[nodeToReplacement[parameter.Key.AsNode()]] = parameter.Value;
            }
        }

        public IEnumerable<SyntaxNodeOrToken> InvocationNodes
            => new SyntaxNodeOrToken[] { MethodInvocationNode }.Concat(ArgumentNodeToMethodParameter.Keys);
    }

    public readonly struct MethodDeclarationInformation
    {
        public readonly IMethodSymbol DeclaredMethodSymbol;
        public readonly SyntaxNode MethodDeclarationNode;
        public readonly Dictionary<IParameterSymbol, SyntaxToken> MethodParameterNodes;

        public MethodDeclarationInformation(
            IMethodSymbol declaredMethodSymbol,
            SyntaxNode methodDeclarationNode,
            Dictionary<IParameterSymbol, SyntaxToken> methodParameterNodes)
        {
            DeclaredMethodSymbol = declaredMethodSymbol;
            MethodDeclarationNode = methodDeclarationNode;
            MethodParameterNodes = methodParameterNodes;
        }
    }

    public sealed class SourceGraphElementsComparer
        : IEqualityComparer<Edge<SyntaxNodeOrToken, SourceGraphEdge>>,
          IEqualityComparer<SyntaxNodeOrToken>
    {
        bool IEqualityComparer<Edge<SyntaxNodeOrToken, SourceGraphEdge>>.Equals(Edge<SyntaxNodeOrToken, SourceGraphEdge> x, Edge<SyntaxNodeOrToken, SourceGraphEdge> y)
            => x.Label == y.Label
                && ((IEqualityComparer<SyntaxNodeOrToken>)this).Equals(x.Source, y.Source)
                && ((IEqualityComparer<SyntaxNodeOrToken>)this).Equals(x.Target, y.Target);

        bool IEqualityComparer<SyntaxNodeOrToken>.Equals(SyntaxNodeOrToken x, SyntaxNodeOrToken y)
            => x.Equals(y);

        int IEqualityComparer<Edge<SyntaxNodeOrToken, SourceGraphEdge>>.GetHashCode(Edge<SyntaxNodeOrToken, SourceGraphEdge> edge)
            => 29 * edge.Source.GetHashCode() + 31 * edge.Target.GetHashCode() + 37 * edge.Label.GetHashCode();

        int IEqualityComparer<SyntaxNodeOrToken>.GetHashCode(SyntaxNodeOrToken node)
            => node.GetHashCode();
    }

    public class SourceGraph : DirectedGraph<SyntaxNodeOrToken, SourceGraphEdge>
    {
        public const string NOTYPE_NAME = "NOTYPE";

        public readonly SemanticModel SemanticModel;
        private readonly ConcurrentDictionary<SyntaxNodeOrToken, string> _nodeToTypeStringCache;
        private readonly HashSet<SyntaxNodeOrToken> _variableDeclarationNodes;
        private readonly HashSet<SyntaxNodeOrToken> _variableUseNodes;
        private readonly HashSet<SyntaxNodeOrToken> _variableWriteNodes;
        private readonly List<MethodInvocationInformation> _methodInvocationSites;
        private readonly List<MethodDeclarationInformation> _methodDeclarationSites;

        private SourceGraph(SemanticModel semanticModel, SourceGraphElementsComparer comparer) : base(comparer, comparer)
        {
            SemanticModel = semanticModel;
            _nodeToTypeStringCache = new ConcurrentDictionary<SyntaxNodeOrToken, string>();
            _variableUseNodes = new HashSet<SyntaxNodeOrToken>();
            _variableWriteNodes = new HashSet<SyntaxNodeOrToken>();
            _variableDeclarationNodes = new HashSet<SyntaxNodeOrToken>();
            _methodInvocationSites = new List<MethodInvocationInformation>();
            _methodDeclarationSites = new List<MethodDeclarationInformation>();
        }

        private SourceGraph(SourceGraph baseGraph, SourceGraphElementsComparer comparer) : base(comparer, comparer)
        {
            SemanticModel = baseGraph.SemanticModel;
            _nodeToTypeStringCache = baseGraph._nodeToTypeStringCache;
            _variableUseNodes = new HashSet<SyntaxNodeOrToken>(baseGraph._variableUseNodes);
            _variableWriteNodes = new HashSet<SyntaxNodeOrToken>(baseGraph._variableWriteNodes);
            _variableDeclarationNodes = new HashSet<SyntaxNodeOrToken>(baseGraph._variableDeclarationNodes);
            _methodInvocationSites = new List<MethodInvocationInformation>(baseGraph._methodInvocationSites);
            _methodDeclarationSites = new List<MethodDeclarationInformation>(baseGraph._methodDeclarationSites);
        }

        public static (SourceGraph, IReadOnlyDictionary<SyntaxNode, SyntaxNodeOrToken> nodeReplacementMap) Create(SemanticModel semanticModel)
        {
            var comparer = new SourceGraphElementsComparer();
            var graph = new SourceGraph(semanticModel, comparer);

            var tree = semanticModel.SyntaxTree;
            var allTokens = tree.GetRoot().DescendantTokens().Where(t => t.Text.Length > 0).ToArray();

            ASTGraphBuilder.AddASTGraph(graph);
            VariableUseGraphBuilder.AddVariableUseGraph(graph);
            DataFlowGraphBuilder.AddDataFlowGraph(graph);
            try
            {
                GuardedByGraphBuilder.AddGuardedByGraph(graph);
            }
            catch (Exception e)
            {
                Console.WriteLine("Failed to add guarded-by edges. Error message: " + e.Message);
            }
            AddNextLexicalUseGraph(graph);
            ReturnToGraphBuilder.AddReturnToGraph(graph);
            MethodUseInformationCollector.AddMethodUseInformation(graph);

            var (compressedGraph, nodeToCompressionReplacement) = graph.GetCompressedGraph();

            foreach (var invocationSite in graph.MethodInvocationSites)
            {
                invocationSite.ReplaceSyntaxNodes(nodeToCompressionReplacement);
            }

            return (compressedGraph, nodeToCompressionReplacement);
        }

        public static SourceGraph CreateEmptyCopy(SourceGraph baseGraph)
        {
            var comparer = new SourceGraphElementsComparer();
            return new SourceGraph(baseGraph, comparer);
        }
  
        private static void AddNextLexicalUseGraph(SourceGraph graph)
        {
            var allTokens = graph.SemanticModel.SyntaxTree.GetRoot().DescendantTokens().Where(t => t.Text.Length > 0).ToArray();

            var nameToLastUse = new Dictionary<string, SyntaxToken>();
            for (int i = 0; i < allTokens.Length; ++i)
            {
                var curTok = allTokens[i];
                if (graph.VariableUseNodes.Contains(curTok))
                {
                    var curName = curTok.Text;
                    if (nameToLastUse.TryGetValue(curName, out var lastUseTok))
                    {
                        graph.AddEdge(curTok, SourceGraphEdge.LastLexicalUse, lastUseTok);
                    }
                    nameToLastUse[curName] = curTok;
                }
            }
        }

        public bool IsVariableDeclaration(SyntaxNodeOrToken node) => _variableDeclarationNodes.Contains(node);

        public bool IsVariableWrite(SyntaxNodeOrToken node) => _variableWriteNodes.Contains(node);

        public ISet<SyntaxNodeOrToken> VariableDeclarationNodes => _variableDeclarationNodes;

        public ISet<SyntaxNodeOrToken> VariableUseNodes => _variableUseNodes;

        public ISet<SyntaxNodeOrToken> VariableWriteNodes => _variableWriteNodes;

        public IList<MethodDeclarationInformation> MethodDeclarationSites => _methodDeclarationSites;

        public IList<MethodInvocationInformation> MethodInvocationSites => _methodInvocationSites;

        public string GetNodeLabel(SyntaxNodeOrToken node, Dictionary<SyntaxNodeOrToken, string> nodeLabelOverrides, bool addTypeToLabel=false)
        {
            if (nodeLabelOverrides.TryGetValue(node, out var label))
            {
                return label;
            }

            if (node.IsToken)
            {
                label = node.ToString();
                // Strip "@" off of masked keywords, as we use proper symbols in the rest of the code, which don't have the @ either:
                if (label.StartsWith("@"))
                {
                    label = label.Substring(1);
                }
            }
            else
            {
                label = node.AsNode().Kind().ToString();
            }

            if (SemanticModel != null)
            {
                var type = GetNodeType(node);
                return label + (addTypeToLabel?" [" + type + "]":"");
            }

            return label;
        }

        private Action<JsonWriter, Dictionary<SyntaxNodeOrToken, int>> WriteNodeLabelsJson(Dictionary<SyntaxNodeOrToken, string> nodeLabelOverrides)
            => (jWriter, nodeNumberer) =>
            {
                jWriter.WritePropertyName("NodeLabels");

                jWriter.WriteStartObject();
                int i = 0;
                foreach ((var node, var nodeNumber) in nodeNumberer.OrderBy(kv=>kv.Value))
                {
                    Debug.Assert(nodeNumber == i++);
                    jWriter.WritePropertyName(nodeNumber.ToString());
                    jWriter.WriteValue(GetNodeLabel(node, nodeLabelOverrides));
                }
                jWriter.WriteEndObject();
            };

        private string GetNodeType(SyntaxNodeOrToken node)
        {
            if (!_nodeToTypeStringCache.TryGetValue(node, out var res))
            {
                // Handle some literal types:
                if (node.IsKind(SyntaxKind.StringLiteralToken))
                {
                    res = "string";
                }
                else if (node.IsKind(SyntaxKind.CharacterLiteralToken))
                {
                    res = "char";
                }
                else if (node.IsKind(SyntaxKind.NumericLiteralToken))
                {
                    res = node.AsToken().Value.GetType().Name.ToLower();
                }
                else
                {
                    var syntaxNode = node.IsNode ? node.AsNode() : node.AsToken().Parent;
                    if (syntaxNode != null)
                    {
                        ISymbol symbol = RoslynUtils.GetReferenceSymbol(syntaxNode, this.SemanticModel);
                        res = GetTypeNameOfSymbol(symbol);
                    }
                    else
                    {
                        res = NOTYPE_NAME;
                    }
                }
                _nodeToTypeStringCache[node] = res;
            }
            return res;
        }

        public static string GetTypeNameOfSymbol(ISymbol symbol)
        {
            string res;
            if (RoslynUtils.GetTypeSymbol(symbol, out var typeSymbol))
            {
                res = typeSymbol.ToString();
            }
            else
            {
                res = NOTYPE_NAME;
            }
            return res;
        }

        public void OverrideTypeForNode(SyntaxNodeOrToken node, string type)
        {
            _nodeToTypeStringCache[node] = type;
        }

        private void WriteNodeTypesJson(JsonWriter jWriter, Dictionary<SyntaxNodeOrToken, int> nodeNumberer)
        {
            jWriter.WritePropertyName("NodeTypes");

            jWriter.WriteStartObject();
            foreach ((var node, var nodeNumber) in nodeNumberer)
            {
                var nodeType = GetNodeType(node);
                if (nodeType != NOTYPE_NAME)
                {
                    jWriter.WritePropertyName(nodeNumber.ToString());
                    jWriter.WriteValue(nodeType);
                }
            }
            jWriter.WriteEndObject();
        }

        private void WriteInvocationInformationJson(JsonWriter jWriter, Dictionary<SyntaxNodeOrToken, int> nodeNumberer)
        {
            jWriter.WritePropertyName("InvocationInformation");

            jWriter.WriteStartObject();
            foreach (var invocationInformation in MethodInvocationSites)
            {
                string invokedMethodFQN = MethodUtils.MethodFullyQualifiedName(invocationInformation.InvokedMethodSymbol);
                foreach (var (argumentNode, parameterSymbol) in invocationInformation.ArgumentNodeToMethodParameter)
                {
                    // Not all call sites are necessarily fully covered by the considered subgraph:
                    if (nodeNumberer.TryGetValue(argumentNode, out int argumentNodeId))
                    {
                        jWriter.WritePropertyName(argumentNodeId.ToString());
                        jWriter.WriteStartArray();
                        jWriter.WriteValue(invokedMethodFQN);
                        jWriter.WriteValue(parameterSymbol.parameterName);
                        jWriter.WriteEndArray();
                    }
                }
                if (nodeNumberer.TryGetValue(invocationInformation.MethodInvocationNode, out int returnNodeId))
                {
                    jWriter.WritePropertyName(returnNodeId.ToString());
                    jWriter.WriteStartArray();
                    jWriter.WriteValue(invokedMethodFQN);
                    jWriter.WriteValue("%RETURN%");
                    jWriter.WriteEndArray();
                }
            }
            jWriter.WriteEndObject();
        }

        public void WriteJson(JsonWriter jWriter, Dictionary<SyntaxNodeOrToken, int> nodeNumberer, Dictionary<SyntaxNodeOrToken, string> nodeLabelOverrides)
            => WriteJson(jWriter,
                         nodeNumberer,
                         new Action<JsonWriter, Dictionary<SyntaxNodeOrToken, int>>[] {
                             WriteNodeLabelsJson(nodeLabelOverrides),
                             WriteNodeTypesJson,
                             WriteInvocationInformationJson,
                         });

        public void ToDotFile(string outputPath, Dictionary<SyntaxNodeOrToken, int> nodeNumberer, Dictionary<SyntaxNodeOrToken, string> nodeLabeler,
            bool diffable = false, string preambleComment = null)
        {
            object ToLineSpan(SyntaxNodeOrToken t)
            {
                var span = t.GetLocation().GetMappedLineSpan();
                return $"{span.StartLinePosition} -- {span.EndLinePosition}";
            }

            ToDotFile(outputPath, diffable ? nodeNumberer.ToDictionary(kv => kv.Key, kv => ToLineSpan(kv.Key))
                                           : nodeNumberer.ToDictionary(kv => kv.Key, kv => (object)kv.Value),
                      node => node.IsToken ? "rectangle" : "circle",
                      node => GetNodeLabel(node, nodeLabeler, addTypeToLabel: true).Replace("\"", "\\\""),
                      preambleComment:preambleComment);
        }

        #region GraphCompression
        /// <summary>
        /// Compute a mapping from nodes that we want to remove to syntax tokens that serve as replacements.
        /// </summary>
        /// <returns></returns>
        private Dictionary<SyntaxNode, SyntaxNodeOrToken> ComputeNodeCompressionMap()
        {
            var nodeToTokenMap = new Dictionary<SyntaxNode, SyntaxNodeOrToken>();

            foreach (var node in Nodes.Where(node => node.IsNode))
            {
                var syntaxNode = node.AsNode();
                SyntaxNodeOrToken? replacementNode = null;
                switch (syntaxNode)
                {
                    case PropertyDeclarationSyntax propDecl:
                        replacementNode = propDecl.Identifier;
                        break;
                    case VariableDeclaratorSyntax varDecl:
                        replacementNode = varDecl.Identifier;
                        break;
                    case SingleVariableDesignationSyntax singleVarSyntax:
                        replacementNode = singleVarSyntax.Identifier;
                        break;
                    case IdentifierNameSyntax idNameSyn:
                        replacementNode = idNameSyn.Identifier;
                        break;
                    case OmittedArraySizeExpressionSyntax ommittedSyn:
                        replacementNode = ommittedSyn.OmittedArraySizeExpressionToken;
                        break;
                    case PredefinedTypeSyntax predefTypeSyn:
                        replacementNode = predefTypeSyn.Keyword;
                        break;
                    case LiteralExpressionSyntax litSyn:
                        replacementNode = litSyn.Token;
                        break;
                    case EnumMemberDeclarationSyntax enumMemberSym:
                        replacementNode = enumMemberSym.Identifier;
                        break;
                    case SimpleBaseTypeSyntax simpleTypeSyntax:
                        replacementNode = simpleTypeSyntax.Type;
                        break;
                    case TypeParameterSyntax typeParSyntax:
                        replacementNode = typeParSyntax.Identifier;
                        break;
                    case BaseExpressionSyntax baseExprSyntax:
                        replacementNode = baseExprSyntax.Token;
                        break;
                    case ThisExpressionSyntax thisSyntax:
                        replacementNode = thisSyntax.Token;
                        break;
                    case ClassOrStructConstraintSyntax classOrStructSyntax:
                        replacementNode = classOrStructSyntax.ClassOrStructKeyword;
                        break;
                    case InterpolatedStringTextSyntax interpolStringSyntax:
                        replacementNode = interpolStringSyntax.TextToken;
                        break;
                    case ArgumentSyntax argSyn:
                        if (argSyn.NameColon == null && argSyn.RefOrOutKeyword.SyntaxTree == null)
                        {
                            replacementNode = argSyn.Expression;
                        }
                        break;
                }

                if (replacementNode.HasValue)
                {
                    nodeToTokenMap[syntaxNode] = replacementNode.Value;
                }
            }
            return nodeToTokenMap;
        }

        public (SourceGraph compressedGraph, IReadOnlyDictionary<SyntaxNode, SyntaxNodeOrToken> nodeToReplacement) GetCompressedGraph()
        {
            var compressedGraph = CreateEmptyCopy(this);
            var nodeToReplacement = ComputeNodeCompressionMap();

            foreach (var sourceNode in Nodes)
            {
                var newSourceNode = sourceNode;
                if (sourceNode.IsNode)
                {
                    if (nodeToReplacement.TryGetValue(sourceNode.AsNode(), out var replacementSourceNode))
                    {
                        newSourceNode = replacementSourceNode;
                    }
                }

                foreach (var edge in GetOutEdges(sourceNode))
                {
                    SyntaxNodeOrToken newTargetNode = edge.Target;
                    if (edge.Target.IsNode)
                    {
                        var targetSyntaxNode = edge.Target.AsNode();
                        while (targetSyntaxNode != null && nodeToReplacement.TryGetValue(targetSyntaxNode, out SyntaxNodeOrToken replacementTargetNode))
                        {
                            if (replacementTargetNode.IsNode)
                            {
                                targetSyntaxNode = replacementTargetNode.AsNode();
                            }
                            else
                            {
                                targetSyntaxNode = null;
                            }
                            newTargetNode = replacementTargetNode;
                        }
                    }

                    //Don't make links between replaced nodes into cycles:
                    if (newSourceNode == newTargetNode && sourceNode != edge.Target)
                    {
                        continue;
                    }
                    compressedGraph.AddEdge(newSourceNode, edge.Label, newTargetNode);
                }
            }

            return (compressedGraph, nodeToReplacement);
        }

        #endregion
    }
}
