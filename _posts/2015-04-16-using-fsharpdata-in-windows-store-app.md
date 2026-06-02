---
layout: post
title: "Using FSharp.Data in Windows Store app"
date: 2015-04-16
categories: [Programming, "F#"]
tags: [F#, FSharp.Data, Microsoft.Net.Http, MVVM, PCL, Windows Store]
original_url: "https://alekseyv-dev.blogspot.com/2015/04/using-fsharpdata-in-windows-store-app.html"
---

Idea to create a dev blog crossed my mind long time ago, but I didn't get to finally do it until now. Hope someone will find my thoughts and experience useful.  
  
Today I want to share my experience of integrating [FSharp.Data](http://fsharp.data/) library into Windows Store application. It took couple of evenings to finally make it work, and few times I thought about dropping this idea and doing everything in good old C# instead. Glad that it finally worked out. So I'd like to share my findings in case someone wants to build something similar.  
  
I started to learn F# in 2011, and the more I got to know it, the more I liked it. Since then I'm trying to use it whenever possible, but unfortunately can't say that I had a lot of success with that.  
  
Recently I decided to rewrite part of  Window Store application I develop as a hobby project, and decided to give F# a try here. Application has HTML parsing component implemented in C# using [HTML Agility Pack](https://www.nuget.org/packages/HtmlAgilityPack). Given that there is an amazing [FSharp.Data](http://fsharp.github.io/FSharp.Data/library/HtmlProvider.html) library with HTML provider, it looked promising to rewrite that code in F#.  
  
My first disappointment was that Windows Store does not support F# natively. I couldn't believe that Microsoft allows building Windows Store apps using C++ and JavaScript, but at the same time F# code needs to go into portable class library (PCL). Regardless of all statements about growing F# share and predictions of its bright future this fact makes it look like a niche language in .NET family. It looks even more strange taking into account that Apple have chosen to go functional with Swift, but that is another story. Hope that Microsoft will start taking F# more seriously in the future.  
  
Luckily there is a great blog post describing how to integrate F# into Windows Store application: <http://ps-a.blogspot.com/2013/01/windows-store-apps-with-f-part-1-make.html>. It helped to setup project and understand how all pieces work together. In the first approach I tried to take maximum advantage out of F# and attempted to reimplement ViewModel. It didn't work nicely. First problem was that ViewModel concept is all about state and .NET events. All that code looks clumsy in F#. Second problem was that I have quite a bit of helper code and ViewModel base class in C# project. It couldn't be referenced from F# unless that code is moved to another C# portable library. That option didn't appeal to me much, so I decided to drop this idea. It might work better in case all ViewModels and related logic in the project are implemented in F#. In fact http://reactiveui.net/ looks like a perfect match here. Unfortunately there is no information available describing how it works with F#. I plan to try it out the next time I build Windows Store app.  
  
So, eventually I decided to keep ViewModel in C# and only port library responsible for HTML parsing and data processing to F#. First I had to figure out how to use FSharp.Data APIs from F# interactive. I installed nuget package, and started exploring.  

```console
Install-Package FSharp.Data
```

It turned out that nuget package contains 2 folders: net40 and portable-net40+sl5+wp8+win8.

net40 is used for native .NET applications, and portable-net40+sl5+wp8+win8 for platforms which work with portable class library including Windows 8. Therefore we'll refer to  
packages\FSharp.Data.2.1.0\lib\net40\FSharp.Data.dll in F# interactive and add reference to packages\FSharp.Data.2.1.0\lib\portable-net40+sl5+wp8+win8\FSharp.Data.dll in Windows Store project.  
  

```fsharp
#r @"C:\Projects\FsharpWindowsStorePrototype\packages\FSharp.Data.2.1.0\lib\net40\FSharp.Data.dll"

open System
open FSharp.Data

type SampleHtmlProvider = HtmlProvider<"http://www.weather.com/weather/today/l/98033:4:US">
let data = SampleHtmlProvider.Load("http://www.weather.com/weather/today/l/98033:4:US")
let info = data.Tables.Table1.Rows.[0].Column1

open System
open FSharp.Data

type SampleHtmlProvider = HtmlProvider<"http://www.weather.com/weather/today/l/98033:4:US">
let data = SampleHtmlProvider.Load("http://www.weather.com/weather/today/l/98033:4:US")
let info = data.Tables.Table1.Rows.[0].Column1
```

namespace System

namespace FSharp

namespace FSharp.Data

type SampleHtmlProvider =  
  class  
  end  
  
Full name: Snippet.SampleHtmlProvider

val data : 'a

val info : 'a  
  
Ok, it looks like provider works, but it doesn't do very good job in providing strongly typed wrapper around HTML document structure. It exposes basic APIs for iterating over document nodes and doing matching using predicate function. This is just enough in most of cases.  
  
Next I added sample module file, referenced F# project from C# Windows Store application and tried to compile it. It didn't work like that:  
  

```text
2>C:\Program Files (x86)\MSBuild\Microsoft\VisualStudio\v12.0\AppxPackage\Microsoft.AppXPackage.Targets(852,9): error MSB3816: Loading assembly "C:\Projects\FsharpWindowsStorePrototype\FsharpPortableLibrary\bin\Debug\FSharp.Data.dll" failed. System.IO.FileNotFoundException: Could not load file or assembly 'FSharp.Core, Version=2.3.5.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a' or one of its dependencies. The system cannot find the file specified.

2>C:\Program Files (x86)\MSBuild\Microsoft\VisualStudio\v12.0\AppxPackage\Microsoft.AppXPackage.Targets(852,9): error MSB3816: File name: 'FSharp.Core, Version=2.3.5.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a'
```

After some googling I found out that FSharp.Data doesn't support PCL [profile 7](http://stackoverflow.com/questions/20332046/correct-version-of-fsharp-core). It turned out that there are multiple PCL profiles to target different set of platforms. I spent few hours trying to change profile type to 47. Solution was unexpected, but straightforward:  
  
1. Use legacy Project Library template for .NET Framework 4.0  

![](/assets/images/using-fsharpdata-in-windows-store-app/image-1.png)

  
  
2. In project settings change target F# runtime to 3.0 (FSharp.Core 2.3.5.0)  

![](/assets/images/using-fsharpdata-in-windows-store-app/image-2.png)

  
  
Finally I could build solution and do HTTP requests ... almost.  
  
It turned out that web server that hosts site that I'm scraping requires UserAgent to be set on HTTP request. There is a known issue with [UserAgent header](http://stackoverflow.com/questions/7454301/is-it-possible-to-modify-the-user-agent-for-a-winrt-httpwebrequest) in Windows Store apps. It is just not possible to set UserAgent header on HttpWebRequest. FSharp.Data provides nice Http helper utility, and I had a hope that it would work:  
  

```fsharp
open System
open FSharp.Data
open FSharp.Data.HttpRequestHeaders

// Run the HTTP web request
Http.RequestString
  ( "https://www.google.com/search?q=super",
    query   = [ "q", "super" ],
    headers = [ Accept HttpContentTypes.Any; UserAgent "Mozilla/5.0" ])
```

namespace System

namespace FSharp

namespace FSharp.Data

  
And it does work ... but only in F# interactive which uses .NET version of the binary. I thought I'm doing something wrong until I found this code in FSharp.Data souces:  
  

```fsharp
#if FX_NO_WEBREQUEST_USERAGENT
            | "user-agent" -> if not (req?UserAgent <- value) then try req.Headers.[HeaderEnum.UserAgent] <- value with _ -> ()
#else
            | "user-agent" -> req.UserAgent <- value
#endif
```

  
This explained everything. Therefore the only option left was to use Microsoft Client HTTP [library](https://www.nuget.org/packages/Microsoft.Net.Http).  
  
Here is how my helper method looks like:

```fsharp
open System
open FSharp.Data
open System.Net
open System.IO
open System.Net.Http
open System.Diagnostics
open System.Text.RegularExpressions

module DownloadHelpers =

    type DataOrError = 
    | Error of Error
    | Data of String

    let buildRequest(url:String, httpMethod:String) = 
        let request = new HttpRequestMessage(new HttpMethod(httpMethod), url)
        request.Headers.UserAgent.ParseAdd("(compatible; MSIE 10.6; Windows NT 6.1; Trident/5.0; InfoPath.2; SLCC1; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729; .NET CLR 2.0.50727) 3gpp-gba UNTRUSTED/1.0");
        request

    let downloadPageByRequest(request:HttpRequestMessage) = async {
        let handler = new HttpClientHandler()
        handler.UseCookies <- false
        let client = new HttpClient(handler)
        try
            use! response = Async.AwaitTask <| client.SendAsync(request)
            let resultCode = response.StatusCode.ToString();
            let! html = Async.AwaitTask <| response.Content.ReadAsStringAsync();
            let processedHtml = Regex.Replace(html, @"\</\d+\>", "") // F# data bug
            return Data processedHtml 
        with ex -> 
                Debug.WriteLine(ex.ToString())
                return Error (Error.Exception ex)
    }

    let downloadPage(url:System.String) = async {
        return! downloadPageByRequest(buildRequest(url, HttpMethod.Get))
    }
```

namespace System

namespace FSharp

namespace System.Data

namespace System.Net

namespace System.IO

namespace System.Diagnostics

namespace System.Text

namespace System.Text.RegularExpressions

module DownloadHelpers  
  
from Snippet

type DataOrError =  
  | Error of obj  
  | Data of String  
  
Full name: Snippet.DownloadHelpers.DataOrError  
  
  type: DataOrError  
  implements: IEquatable<DataOrError>  
  implements: Collections.IStructuralEquatable

union case DataOrError.Error: obj -> DataOrError

Multiple items  
union case DataOrError.Data: String -> DataOrError  
  
--------------------  
namespace System.Data

type String =  
  class  
    new : char -> string  
    new : char \* int \* int -> string  
    new : System.SByte -> string  
    new : System.SByte \* int \* int -> string  
    new : System.SByte \* int \* int \* System.Text.Encoding -> string  
    new : char [] \* int \* int -> string  
    new : char [] -> string  
    new : char \* int -> string  
    member Chars : int -> char  
    member Clone : unit -> obj  
    member CompareTo : obj -> int  
    member CompareTo : string -> int  
    member Contains : string -> bool  
    member CopyTo : int \* char [] \* int \* int -> unit  
    member EndsWith : string -> bool  
    member EndsWith : string \* System.StringComparison -> bool  
    member EndsWith : string \* bool \* System.Globalization.CultureInfo -> bool  
    member Equals : obj -> bool  
    member Equals : string -> bool  
    member Equals : string \* System.StringComparison -> bool  
    member GetEnumerator : unit -> System.CharEnumerator  
    member GetHashCode : unit -> int  
    member GetTypeCode : unit -> System.TypeCode  
    member IndexOf : char -> int  
    member IndexOf : string -> int  
    member IndexOf : char \* int -> int  
    member IndexOf : string \* int -> int  
    member IndexOf : string \* System.StringComparison -> int  
    member IndexOf : char \* int \* int -> int  
    member IndexOf : string \* int \* int -> int  
    member IndexOf : string \* int \* System.StringComparison -> int  
    member IndexOf : string \* int \* int \* System.StringComparison -> int  
    member IndexOfAny : char [] -> int  
    member IndexOfAny : char [] \* int -> int  
    member IndexOfAny : char [] \* int \* int -> int  
    member Insert : int \* string -> string  
    member IsNormalized : unit -> bool  
    member IsNormalized : System.Text.NormalizationForm -> bool  
    member LastIndexOf : char -> int  
    member LastIndexOf : string -> int  
    member LastIndexOf : char \* int -> int  
    member LastIndexOf : string \* int -> int  
    member LastIndexOf : string \* System.StringComparison -> int  
    member LastIndexOf : char \* int \* int -> int  
    member LastIndexOf : string \* int \* int -> int  
    member LastIndexOf : string \* int \* System.StringComparison -> int  
    member LastIndexOf : string \* int \* int \* System.StringComparison -> int  
    member LastIndexOfAny : char [] -> int  
    member LastIndexOfAny : char [] \* int -> int  
    member LastIndexOfAny : char [] \* int \* int -> int  
    member Length : int  
    member Normalize : unit -> string  
    member Normalize : System.Text.NormalizationForm -> string  
    member PadLeft : int -> string  
    member PadLeft : int \* char -> string  
    member PadRight : int -> string  
    member PadRight : int \* char -> string  
    member Remove : int -> string  
    member Remove : int \* int -> string  
    member Replace : char \* char -> string  
    member Replace : string \* string -> string  
    member Split : char [] -> string []  
    member Split : char [] \* int -> string []  
    member Split : char [] \* System.StringSplitOptions -> string []  
    member Split : string [] \* System.StringSplitOptions -> string []  
    member Split : char [] \* int \* System.StringSplitOptions -> string []  
    member Split : string [] \* int \* System.StringSplitOptions -> string []  
    member StartsWith : string -> bool  
    member StartsWith : string \* System.StringComparison -> bool  
    member StartsWith : string \* bool \* System.Globalization.CultureInfo -> bool  
    member Substring : int -> string  
    member Substring : int \* int -> string  
    member ToCharArray : unit -> char []  
    member ToCharArray : int \* int -> char []  
    member ToLower : unit -> string  
    member ToLower : System.Globalization.CultureInfo -> string  
    member ToLowerInvariant : unit -> string  
    member ToString : unit -> string  
    member ToString : System.IFormatProvider -> string  
    member ToUpper : unit -> string  
    member ToUpper : System.Globalization.CultureInfo -> string  
    member ToUpperInvariant : unit -> string  
    member Trim : unit -> string  
    member Trim : char [] -> string  
    member TrimEnd : char [] -> string  
    member TrimStart : char [] -> string  
    static val Empty : string  
    static member Compare : string \* string -> int  
    static member Compare : string \* string \* bool -> int  
    static member Compare : string \* string \* System.StringComparison -> int  
    static member Compare : string \* string \* System.Globalization.CultureInfo \* System.Globalization.CompareOptions -> int  
    static member Compare : string \* string \* bool \* System.Globalization.CultureInfo -> int  
    static member Compare : string \* int \* string \* int \* int -> int  
    static member Compare : string \* int \* string \* int \* int \* bool -> int  
    static member Compare : string \* int \* string \* int \* int \* System.StringComparison -> int  
    static member Compare : string \* int \* string \* int \* int \* bool \* System.Globalization.CultureInfo -> int  
    static member Compare : string \* int \* string \* int \* int \* System.Globalization.CultureInfo \* System.Globalization.CompareOptions -> int  
    static member CompareOrdinal : string \* string -> int  
    static member CompareOrdinal : string \* int \* string \* int \* int -> int  
    static member Concat : obj -> string  
    static member Concat : obj [] -> string  
    static member Concat<'T> : System.Collections.Generic.IEnumerable<'T> -> string  
    static member Concat : System.Collections.Generic.IEnumerable<string> -> string  
    static member Concat : string [] -> string  
    static member Concat : obj \* obj -> string  
    static member Concat : string \* string -> string  
    static member Concat : obj \* obj \* obj -> string  
    static member Concat : string \* string \* string -> string  
    static member Concat : obj \* obj \* obj \* obj -> string  
    static member Concat : string \* string \* string \* string -> string  
    static member Copy : string -> string  
    static member Equals : string \* string -> bool  
    static member Equals : string \* string \* System.StringComparison -> bool  
    static member Format : string \* obj -> string  
    static member Format : string \* obj [] -> string  
    static member Format : string \* obj \* obj -> string  
    static member Format : System.IFormatProvider \* string \* obj [] -> string  
    static member Format : string \* obj \* obj \* obj -> string  
    static member Intern : string -> string  
    static member IsInterned : string -> string  
    static member IsNullOrEmpty : string -> bool  
    static member IsNullOrWhiteSpace : string -> bool  
    static member Join : string \* string [] -> string  
    static member Join : string \* obj [] -> string  
    static member Join<'T> : string \* System.Collections.Generic.IEnumerable<'T> -> string  
    static member Join : string \* System.Collections.Generic.IEnumerable<string> -> string  
    static member Join : string \* string [] \* int \* int -> string  
  end  
  
Full name: System.String  
  
  type: String  
  implements: IComparable  
  implements: ICloneable  
  implements: IConvertible  
  implements: IComparable<string>  
  implements: seq<char>  
  implements: Collections.IEnumerable  
  implements: IEquatable<string>

val buildRequest : String \* String -> 'a  
  
Full name: Snippet.DownloadHelpers.buildRequest

val url : String  
  
  type: String  
  implements: IComparable  
  implements: ICloneable  
  implements: IConvertible  
  implements: IComparable<string>  
  implements: seq<char>  
  implements: Collections.IEnumerable  
  implements: IEquatable<string>

val httpMethod : String  
  
  type: String  
  implements: IComparable  
  implements: ICloneable  
  implements: IConvertible  
  implements: IComparable<string>  
  implements: seq<char>  
  implements: Collections.IEnumerable  
  implements: IEquatable<string>

val request : 'a

val downloadPageByRequest : 'a -> Async<DataOrError>  
  
Full name: Snippet.DownloadHelpers.downloadPageByRequest

val async : AsyncBuilder  
  
Full name: Microsoft.FSharp.Core.ExtraTopLevelOperators.async

val handler : 'a

val client : 'a

val response : 'a (requires 'a :> IDisposable)  
  
  type: 'a  
  implements: IDisposable

Multiple items  
type Async<'T>  
  
Full name: Microsoft.FSharp.Control.Async<\_>  
  
--------------------  
type Async  
with  
  static member AsBeginEnd : computation:('Arg -> Async<'T>) -> ('Arg \* AsyncCallback \* obj -> IAsyncResult) \* (IAsyncResult -> 'T) \* (IAsyncResult -> unit)  
  static member AwaitEvent : event:IEvent<'Del,'T> \* ?cancelAction:(unit -> unit) -> Async<'T> (requires delegate and 'Del :> Delegate)  
  static member AwaitIAsyncResult : iar:IAsyncResult \* ?millisecondsTimeout:int -> Async<bool>  
  static member AwaitTask : task:Threading.Tasks.Task<'T> -> Async<'T>  
  static member AwaitWaitHandle : waitHandle:Threading.WaitHandle \* ?millisecondsTimeout:int -> Async<bool>  
  static member CancelDefaultToken : unit -> unit  
  static member Catch : computation:Async<'T> -> Async<Choice<'T,exn>>  
  static member FromBeginEnd : beginAction:(AsyncCallback \* obj -> IAsyncResult) \* endAction:(IAsyncResult -> 'T) \* ?cancelAction:(unit -> unit) -> Async<'T>  
  static member FromBeginEnd : arg:'Arg1 \* beginAction:('Arg1 \* AsyncCallback \* obj -> IAsyncResult) \* endAction:(IAsyncResult -> 'T) \* ?cancelAction:(unit -> unit) -> Async<'T>  
  static member FromBeginEnd : arg1:'Arg1 \* arg2:'Arg2 \* beginAction:('Arg1 \* 'Arg2 \* AsyncCallback \* obj -> IAsyncResult) \* endAction:(IAsyncResult -> 'T) \* ?cancelAction:(unit -> unit) -> Async<'T>  
  static member FromBeginEnd : arg1:'Arg1 \* arg2:'Arg2 \* arg3:'Arg3 \* beginAction:('Arg1 \* 'Arg2 \* 'Arg3 \* AsyncCallback \* obj -> IAsyncResult) \* endAction:(IAsyncResult -> 'T) \* ?cancelAction:(unit -> unit) -> Async<'T>  
  static member FromContinuations : callback:(('T -> unit) \* (exn -> unit) \* (OperationCanceledException -> unit) -> unit) -> Async<'T>  
  static member Ignore : computation:Async<'T> -> Async<unit>  
  static member OnCancel : interruption:(unit -> unit) -> Async<IDisposable>  
  static member Parallel : computations:seq<Async<'T>> -> Async<'T []>  
  static member RunSynchronously : computation:Async<'T> \* ?timeout:int \* ?cancellationToken:Threading.CancellationToken -> 'T  
  static member Sleep : millisecondsDueTime:int -> Async<unit>  
  static member Start : computation:Async<unit> \* ?cancellationToken:Threading.CancellationToken -> unit  
  static member StartAsTask : computation:Async<'T> \* ?taskCreationOptions:Threading.Tasks.TaskCreationOptions \* ?cancellationToken:Threading.CancellationToken -> Threading.Tasks.Task<'T>  
  static member StartChild : computation:Async<'T> \* ?millisecondsTimeout:int -> Async<Async<'T>>  
  static member StartChildAsTask : computation:Async<'T> \* ?taskCreationOptions:Threading.Tasks.TaskCreationOptions -> Async<Threading.Tasks.Task<'T>>  
  static member StartImmediate : computation:Async<unit> \* ?cancellationToken:Threading.CancellationToken -> unit  
  static member StartWithContinuations : computation:Async<'T> \* continuation:('T -> unit) \* exceptionContinuation:(exn -> unit) \* cancellationContinuation:(OperationCanceledException -> unit) \* ?cancellationToken:Threading.CancellationToken -> unit  
  static member SwitchToContext : syncContext:Threading.SynchronizationContext -> Async<unit>  
  static member SwitchToNewThread : unit -> Async<unit>  
  static member SwitchToThreadPool : unit -> Async<unit>  
  static member TryCancelled : computation:Async<'T> \* compensation:(OperationCanceledException -> unit) -> Async<'T>  
  static member CancellationToken : Async<Threading.CancellationToken>  
  static member DefaultCancellationToken : Threading.CancellationToken  
end  
  
Full name: Microsoft.FSharp.Control.Async

static member Async.AwaitTask : task:Threading.Tasks.Task<'T> -> Async<'T>

val resultCode : 'a

val html : string  
  
  type: string  
  implements: IComparable  
  implements: ICloneable  
  implements: IConvertible  
  implements: IComparable<string>  
  implements: seq<char>  
  implements: Collections.IEnumerable  
  implements: IEquatable<string>

val processedHtml : string  
  
  type: string  
  implements: IComparable  
  implements: ICloneable  
  implements: IConvertible  
  implements: IComparable<string>  
  implements: seq<char>  
  implements: Collections.IEnumerable  
  implements: IEquatable<string>

type Regex =  
  class  
    new : string -> System.Text.RegularExpressions.Regex  
    new : string \* System.Text.RegularExpressions.RegexOptions -> System.Text.RegularExpressions.Regex  
    new : string \* System.Text.RegularExpressions.RegexOptions \* System.TimeSpan -> System.Text.RegularExpressions.Regex  
    member GetGroupNames : unit -> string []  
    member GetGroupNumbers : unit -> int []  
    member GroupNameFromNumber : int -> string  
    member GroupNumberFromName : string -> int  
    member IsMatch : string -> bool  
    member IsMatch : string \* int -> bool  
    member Match : string -> System.Text.RegularExpressions.Match  
    member Match : string \* int -> System.Text.RegularExpressions.Match  
    member Match : string \* int \* int -> System.Text.RegularExpressions.Match  
    member MatchTimeout : System.TimeSpan  
    member Matches : string -> System.Text.RegularExpressions.MatchCollection  
    member Matches : string \* int -> System.Text.RegularExpressions.MatchCollection  
    member Options : System.Text.RegularExpressions.RegexOptions  
    member Replace : string \* string -> string  
    member Replace : string \* System.Text.RegularExpressions.MatchEvaluator -> string  
    member Replace : string \* string \* int -> string  
    member Replace : string \* System.Text.RegularExpressions.MatchEvaluator \* int -> string  
    member Replace : string \* string \* int \* int -> string  
    member Replace : string \* System.Text.RegularExpressions.MatchEvaluator \* int \* int -> string  
    member RightToLeft : bool  
    member Split : string -> string []  
    member Split : string \* int -> string []  
    member Split : string \* int \* int -> string []  
    member ToString : unit -> string  
    static val InfiniteMatchTimeout : System.TimeSpan  
    static member CacheSize : int with get, set  
    static member CompileToAssembly : System.Text.RegularExpressions.RegexCompilationInfo [] \* System.Reflection.AssemblyName -> unit  
    static member CompileToAssembly : System.Text.RegularExpressions.RegexCompilationInfo [] \* System.Reflection.AssemblyName \* System.Reflection.Emit.CustomAttributeBuilder [] -> unit  
    static member CompileToAssembly : System.Text.RegularExpressions.RegexCompilationInfo [] \* System.Reflection.AssemblyName \* System.Reflection.Emit.CustomAttributeBuilder [] \* string -> unit  
    static member Escape : string -> string  
    static member IsMatch : string \* string -> bool  
    static member IsMatch : string \* string \* System.Text.RegularExpressions.RegexOptions -> bool  
    static member IsMatch : string \* string \* System.Text.RegularExpressions.RegexOptions \* System.TimeSpan -> bool  
    static member Match : string \* string -> System.Text.RegularExpressions.Match  
    static member Match : string \* string \* System.Text.RegularExpressions.RegexOptions -> System.Text.RegularExpressions.Match  
    static member Match : string \* string \* System.Text.RegularExpressions.RegexOptions \* System.TimeSpan -> System.Text.RegularExpressions.Match  
    static member Matches : string \* string -> System.Text.RegularExpressions.MatchCollection  
    static member Matches : string \* string \* System.Text.RegularExpressions.RegexOptions -> System.Text.RegularExpressions.MatchCollection  
    static member Matches : string \* string \* System.Text.RegularExpressions.RegexOptions \* System.TimeSpan -> System.Text.RegularExpressions.MatchCollection  
    static member Replace : string \* string \* string -> string  
    static member Replace : string \* string \* System.Text.RegularExpressions.MatchEvaluator -> string  
    static member Replace : string \* string \* string \* System.Text.RegularExpressions.RegexOptions -> string  
    static member Replace : string \* string \* System.Text.RegularExpressions.MatchEvaluator \* System.Text.RegularExpressions.RegexOptions -> string  
    static member Replace : string \* string \* string \* System.Text.RegularExpressions.RegexOptions \* System.TimeSpan -> string  
    static member Replace : string \* string \* System.Text.RegularExpressions.MatchEvaluator \* System.Text.RegularExpressions.RegexOptions \* System.TimeSpan -> string  
    static member Split : string \* string -> string []  
    static member Split : string \* string \* System.Text.RegularExpressions.RegexOptions -> string []  
    static member Split : string \* string \* System.Text.RegularExpressions.RegexOptions \* System.TimeSpan -> string []  
    static member Unescape : string -> string  
  end  
  
Full name: System.Text.RegularExpressions.Regex  
  
  type: Regex  
  implements: Runtime.Serialization.ISerializable

Regex.Replace(input: string, pattern: string, evaluator: MatchEvaluator) : string  
Regex.Replace(input: string, pattern: string, replacement: string) : string  
Regex.Replace(input: string, pattern: string, evaluator: MatchEvaluator, options: RegexOptions) : string  
Regex.Replace(input: string, pattern: string, replacement: string, options: RegexOptions) : string  
Regex.Replace(input: string, pattern: string, evaluator: MatchEvaluator, options: RegexOptions, matchTimeout: TimeSpan) : string  
Regex.Replace(input: string, pattern: string, replacement: string, options: RegexOptions, matchTimeout: TimeSpan) : string

val ex : exn  
  
  type: exn  
  implements: Runtime.Serialization.ISerializable  
  implements: Runtime.InteropServices.\_Exception

type Debug =  
  class  
    static member Assert : bool -> unit  
    static member Assert : bool \* string -> unit  
    static member Assert : bool \* string \* string -> unit  
    static member Assert : bool \* string \* string \* obj [] -> unit  
    static member AutoFlush : bool with get, set  
    static member Close : unit -> unit  
    static member Fail : string -> unit  
    static member Fail : string \* string -> unit  
    static member Flush : unit -> unit  
    static member Indent : unit -> unit  
    static member IndentLevel : int with get, set  
    static member IndentSize : int with get, set  
    static member Listeners : System.Diagnostics.TraceListenerCollection  
    static member Print : string -> unit  
    static member Print : string \* obj [] -> unit  
    static member Unindent : unit -> unit  
    static member Write : string -> unit  
    static member Write : obj -> unit  
    static member Write : string \* string -> unit  
    static member Write : obj \* string -> unit  
    static member WriteIf : bool \* string -> unit  
    static member WriteIf : bool \* obj -> unit  
    static member WriteIf : bool \* string \* string -> unit  
    static member WriteIf : bool \* obj \* string -> unit  
    static member WriteLine : string -> unit  
    static member WriteLine : obj -> unit  
    static member WriteLine : string \* string -> unit  
    static member WriteLine : obj \* string -> unit  
    static member WriteLine : string \* obj [] -> unit  
    static member WriteLineIf : bool \* string -> unit  
    static member WriteLineIf : bool \* obj -> unit  
    static member WriteLineIf : bool \* string \* string -> unit  
    static member WriteLineIf : bool \* obj \* string -> unit  
  end  
  
Full name: System.Diagnostics.Debug

Debug.WriteLine(value: obj) : unit  
Debug.WriteLine(message: string) : unit  
Debug.WriteLine(format: string, args: obj []) : unit  
Debug.WriteLine(value: obj, category: string) : unit  
Debug.WriteLine(message: string, category: string) : unit

Object.ToString() : string

type Exception =  
  class  
    new : unit -> System.Exception  
    new : string -> System.Exception  
    new : string \* System.Exception -> System.Exception  
    member Data : System.Collections.IDictionary  
    member GetBaseException : unit -> System.Exception  
    member GetObjectData : System.Runtime.Serialization.SerializationInfo \* System.Runtime.Serialization.StreamingContext -> unit  
    member GetType : unit -> System.Type  
    member HResult : int with get, set  
    member HelpLink : string with get, set  
    member InnerException : System.Exception  
    member Message : string  
    member Source : string with get, set  
    member StackTrace : string  
    member TargetSite : System.Reflection.MethodBase  
    member ToString : unit -> string  
  end  
  
Full name: System.Exception  
  
  type: Exception  
  implements: Runtime.Serialization.ISerializable  
  implements: Runtime.InteropServices.\_Exception

val downloadPage : String -> Async<DataOrError>  
  
Full name: Snippet.DownloadHelpers.downloadPage

Note that there is an bug in FSharp.Data 2.1.0, so that any closing tag with number like </1> breaks the parser.  
  
Finally everything worked, but can't say that it was the end of the story.  
When I tried to pull project from Git to another computer it refused to build. For some reason F# project was broken. Nuget pulled all dependencies, but for build didn't work.  
Googling and experimenting proved that there is an issue with Microsoft.Bcl dependency which is required by Microsoft.Net.Http. By default Nuget tries to pull older version of this library, while Microsoft.Net.Http works only with the most recent one. The solution which worked for me was to clean Nuget cache, remove Microsoft.Bcl and Microsoft.Bcl.Build projects from packages folder, and request Nuget to install those projects manually before installing Microsoft.Net.Http.  
  

```console
Install-Package Microsoft.Bcl
```

  
It helped to solve dependency issues.  
  
When I did that recently, I also updated FSharp.Data to version 2.2.0. This update broke my application, and now I have to investigate what changed.  
  
Was it worth messing up with F#? Definitely I've spent way more time trying to fix all issues on the way than I'd spend writing same logic in C#. But it was fun and I'm glad that finally everything worked out.
