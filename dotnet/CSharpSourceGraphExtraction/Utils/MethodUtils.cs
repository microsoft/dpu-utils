using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace MSRC.DPU.CSharpSourceGraphExtraction.Utils
{
    public static class MethodUtils
    {
        public static IEnumerable<MethodDeclarationSyntax> GetAllMethodDeclarations(SyntaxNode root)
        {
            return root.DescendantNodes().OfType<MethodDeclarationSyntax>();
        }

        public static (string Summary, string Returns, Dictionary<IParameterSymbol, string> ParameterComments)
            GetDocumentationComment(IMethodSymbol methodSymbol, bool recurseToParents=true)
        {
            var comment = methodSymbol.GetDocumentationCommentXml().Trim();
            
            if (string.IsNullOrWhiteSpace(comment) && recurseToParents)
            {
                foreach (var parentMethod in AllImplementedMethods(methodSymbol))
                {
                    comment = parentMethod.GetDocumentationCommentXml().Trim();
                    if (!string.IsNullOrWhiteSpace(comment))
                    {
                        break;
                    }
                }
            }

            if (string.IsNullOrWhiteSpace(comment))
            {
                return ("", "", null);
            }

            if (!comment.StartsWith("<member"))
            {
                comment = "<member>" + comment + "</member>";
            }
            var xmlDoc = new XmlDocument();
            try
            {
                xmlDoc.LoadXml(comment);
            } catch (Exception)
            {
                return ("", "", null);
            }

            if (xmlDoc.SelectSingleNode("member") == null)
            {
                return ("", "", null);
            }

            var memberXmlNode = xmlDoc.SelectSingleNode("member");

            string summary = "";
            if (memberXmlNode.SelectSingleNode("summary") != null) {
                summary = xmlDoc.SelectSingleNode("member").SelectSingleNode("summary").InnerXml.Trim();
            }

            var parameterComments = new Dictionary<IParameterSymbol, string>();
            var paramNamesToSymbols = methodSymbol.Parameters.ToDictionary(s => s.Name, s => s);

            foreach(var paramXmlNode in memberXmlNode.SelectNodes("param"))
            {
                var paramName = ((XmlNode)paramXmlNode).Attributes["name"].InnerText;
                if (paramNamesToSymbols.ContainsKey(paramName))
                {
                    parameterComments.Add(paramNamesToSymbols[paramName], ((XmlNode)paramXmlNode).InnerXml.Trim());
                }
            }

            string returnVal = "";
            if (memberXmlNode.SelectSingleNode("returns") != null)
            {
                returnVal = xmlDoc.SelectSingleNode("member").SelectSingleNode("returns").InnerXml.Trim();
            }
                
            return (summary, returnVal, parameterComments);            
        }

        public static string MethodFullyQualifiedName(IMethodSymbol methodSymbol)
            => methodSymbol.OriginalDefinition.ToDisplayString(); // Using the OriginalDefinition avoids instantiated type variables.

        public static IEnumerable<IMethodSymbol> AllImplementedMethods(IMethodSymbol methodSymbol)
        {
            var seenMethods = new HashSet<IMethodSymbol>();
            if (methodSymbol.OverriddenMethod != null)
            {
                yield return methodSymbol.OverriddenMethod;
                foreach (var m in AllImplementedMethods(methodSymbol.OverriddenMethod))
                {
                    yield return m;
                    seenMethods.Add(m);
                }
            }

            foreach(var implementedMethod in methodSymbol.ContainingType.AllInterfaces
                .SelectMany(iface => iface.GetMembers().OfType<IMethodSymbol>())
                .Where(m => methodSymbol.Equals(methodSymbol.ContainingType.FindImplementationForInterfaceMember(m))))
            {
                if (seenMethods.Add(implementedMethod))
                {
                    yield return implementedMethod;
                    foreach(var m in AllImplementedMethods(implementedMethod))
                    {
                        yield return m;
                        seenMethods.Add(m);
                    }
                }                
            }
        }

        /// <summary>
        /// Copied from Roslyn source code. Determines the parameter for a given argument
        /// </summary>
        /// <param name="argumentList"></param>
        /// <param name="argument"></param>
        /// <param name="symbol"></param>
        /// <returns></returns>
        public static IParameterSymbol DetermineParameter(BaseArgumentListSyntax argumentList, ArgumentSyntax argument, IMethodSymbol symbol)
        {
            var parameters = symbol.Parameters;

            // Handle named argument
            if (argument.NameColon != null && !argument.NameColon.IsMissing)
            {
                var name = argument.NameColon.Name.Identifier.ValueText;
                return parameters.FirstOrDefault(p => p.Name == name);
            }

            // Handle positional argument
            var index = argumentList.Arguments.IndexOf(argument);
            if (index < 0)
            {
                return null;
            }

            if (index < parameters.Length)
            {
                return parameters[index];
            }

            // Handle Params
            var lastParameter = parameters.LastOrDefault();
            if (lastParameter == null)
            {
                return null;
            }

            if (lastParameter.IsParams)
            {
                return lastParameter;
            }

            return null;
        }
    }
}
