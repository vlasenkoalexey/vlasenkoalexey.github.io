---
layout: post
title: "Implementing F# interactive intellisense support for Visual Studio"
date: 2015-10-04
categories: [Programming, "F#"]
tags: [F#, FSI, Intellisense, Visual Studio, Visual Studio Extensibility]
original_url: "https://alekseyv-dev.blogspot.com/2015/10/implementing-f-interactive-intellisense.html"
---

While playing with F# I found that I'm really missing intellisense support for FSI interactive. It is very useful for exploring APIs and language features. I know, "official" suggestion is to write code in script file where full intellisense is available, and then execute it in FSI. But that is less then optimal, at least for me.  
  
Lack of intellisense is even more surprising given that console version of fsi.exe has some limited support for autocompletion. I even considered embedding console window with running FSI in Visual Studio, but that turned our to be problematic.  
  
I googled problem a bit, and it didn't seem to be very hard to fix it. F# is open sourced now, and clearly there are some APIs for autocomplete in FSI: <https://github.com/fsharp/fsharp/blob/master/src/fsharp/fsiserver/fsiserver.fs>  
  
Adding intellisense support to editor also didn't seem problematic, there is a walkthrough available:  
<https://msdn.microsoft.com/en-us/library/vstudio/ee372314(v=vs.110).aspx>  
  
Then everything I expected to do it taking walkthrough and integrating it with FSI autocompletion APIs. Why should it take longer than couple of evenings? But as usual for software development nothing works as expected from the first attempt.  
  
First I had to find a way to access those FSI APIs. FsiLanguageService is registered as a Visual Studio extensibility service. This is our entry point, everything else can be solved using Reflection. It is obviously a hack and it'll likely break if FSI sources are changed. But it will work, and telling from other's code this is the way extensibility is done for Visual Studio.  
In order to access internal properties and fields using C# dynamic keyword we can use [ExposedObject](http://igoro.com/archive/use-c-dynamic-typing-to-conveniently-access-internals-of-an-object/). Also I had to modify it a bit because it doesn't support fields declared in nested classes. It is a bit ugly, but does its job:  

```csharp
public FsiLanguageServiceHelper()
{
    fsiAssembly = Assembly.Load("FSharp.VS.FSI, Version=12.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a");
    fsiLanguageServiceType = fsiAssembly.GetType("Microsoft.VisualStudio.FSharp.Interactive.FsiLanguageService");
    sessionsType = fsiAssembly.GetType("Microsoft.VisualStudio.FSharp.Interactive.Session.Sessions");
    fsiWindowType = fsiAssembly.GetType(FsiToolWindowClassName);
}
 
private Object GetSession()
{
    try
    {
        var providerGlobal = (IOleServiceProvider)Package.GetGlobalService(typeof(IOleServiceProvider));
        var provider = new ServiceProvider(providerGlobal);
        dynamic fsiLanguageService = ExposedObject.From(provider.GetService(fsiLanguageServiceType));
        dynamic sessions = fsiLanguageService.sessions;
        if (sessions != null)
        { 
            dynamic sessionsValue = ExposedObject.From(sessions).Value;
            dynamic sessionR = ExposedObject.From(sessionsValue).sessionR;
            dynamic sessionRValue = ExposedObject.From(sessionR).Value;
            dynamic sessionRValueValue = ExposedObject.From(sessionRValue).Value;
            dynamic exitedE = ExposedObject.From(sessionRValueValue).exitedE;
 
                MethodInfo methodInfo = fsiAssembly.GetType("Microsoft.VisualStudio.FSharp.Interactive.Session+Session").GetMethod("get_Exited");
                IObservable<EventArgs> exited = methodInfo.Invoke(sessionRValueValue, null);
                IObserver<EventArgs> obsvr = Observer.Create<EventArgs>(
                    x => {
                    //    RegisterAutocompleteServer(); 
                    },
                    ex => { },
                    () => { });
 
                exited.Subscribe(obsvr);
 
        
            return sessionRValueValue;
        }
    }
    catch (Exception ex)
    {
        Debug.WriteLine(ex);
    }
 
    return null;
}
```

The second problem is that for some reason FSI is compiled with #FSI\_SERVER\_INTELLISENSE flag off. It complicates things a lot. There are 3 options here:  
  
1. Recompile FSI with #FSI\_SERVER\_INTELLISENSE flag on and use modified fsi.exe file. I tried this, and it worked. Also I didn't like to use hacked FSI, so decided to move further.  
2. Implement autocomplete APIs on my own.  
3. Try to find an entry point into FSI process, and use reflection to call autocomplete APIs.  
  
It wasn't clear if it is possible to find a way to get into existing FSI process, therefor I decided to start with my own implementation.  
  
FSI interactive binary is designed to be run in 2 modes - as a console application, or as a fsi server. In case of fsi server, std-in and std-out is redirected, and communication with Visual Studio happens over Remoting: <https://github.com/fsharp/fsharp/blob/master/src/fsharp/fsiserver/fsiserver.fs>  
  
For me it meant that I need to run some code within FSI process, and that I need to establish some intra-process communication. First I tried to use Remoting since it is already used by FSI server, and it almost worked. Once in a while it failed with some cryptic errors which have no sense at all. So I replaced it by WCF service operating over NamedPipes, and it worked much better. Here is my service definition:  
  

```fsharp
[<Serializable>]
[<ServiceContract>]
type AutocompleteService = 
    [<OperationContract>]
    abstract Ping : a:unit -> bool
    [<OperationContract>]
    abstract GetBaseDirectory : b:unit -> String
    [<OperationContract>]
    abstract GetCompletions: prefix:String -> providerType:IntellisenseProviderType -> seq<Completion>
```

  
I didn't come up with anything better than adding reference to my dll and then executing command to start my server by piping those commands directly into FSI input. Not perfect solution, but works well.  
  
Implementing basic auto-completion provider wasn't hard. I used reflection APIs to iterate over type system, and a hack similar to the one used in FsEye in order to get access to variable names: <https://github.com/SwensenSoftware/fseye/blob/63266a99e59a5eb70173869a05fe9a3bc4b96466/FsEye/Fsi/SessionQueries.fs>. Basically variables of FSI sessions are stored in dynamic assembly and names start with FSI\_:  
  

```fsharp
let getVariableNames(fsiAssembly:Assembly) =
    fsiAssembly.GetTypes()//FSI types have the name pattern FSI_####, where #### is the order in which they were created
    |> Seq.filter (fun ty -> ty.Name.StartsWith("FSI_"))
    |> Seq.sortBy (fun ty -> ty.Name.Split('_').[1] |> int)
    |> Seq.collect (fun ty -> getPropertyInfosForType(ty))
    |> Seq.map (fun pi -> pi.Name)
    |> Seq.distinct
```

  
I also added namespaces opened in F# by default- Microsoft.FSharp.Collections and Microsoft.FSharp.Core.Printf. It helped a bit with usability, but there was another problem.  
  
Autocomplete popup just rejected processing arrow down button. It is critical for navigation buttons to work for autocomplete popup, so I couldn't leave this issue unsolved. One of the biggest problems with Visual Studio extensibility is lack of good documentation. So the best source of information for me were GitHub sources for similar projects, and debugger. It turned out that CommandHandlers for key events in Visual Studio are organized in a hierarchical manner. If top level CommandHandler processes key event, and marks event as processed, there is nothing we can do on lower level CommandHanlder. In case with FSI window, [fsiToolWindow](https://github.com/Microsoft/visualfsharp/blob/275b832e9dd1a4bd64ed3accd218384b901be1d2/vsintegration/src/vs/FsPkgs/FSharp.VS.FSI/fsiSessionToolWindow.fs#L730) processing arrow up/down keypress events for navigating through history, and therefore intercepted those events. And in fact when cursor was not set to the bottom line of FSI those events worked just fine since FSI doesn't process them for navigating history. I guess I tried every reasonable approach to make it work, but it didn't help. So the only way for me to solve it was to make FSI think that it should not iterate over history. Luckily there is a way:  
  

```csharp
private void SetFsiToolWindowCompletionSetVisibilty(bool displayed)
{
    if (fsiToolWindow != null)
    {
        dynamic source = ExposedObject.From(fsiToolWindow).source;
        if (source != null)
        {
            dynamic completionSet = ExposedObject.From(source).CompletionSet;
            if (completionSet != null)
            {
                dynamic completionSetExp = ExposedObject.From(completionSet);
                completionSetExp.displayed = displayed;
            }
        }
    }
}
```

  
Finally everything worked, but after playing with this solution for some time I realized that it is far from perfect.  
  
So I got back to original idea - find a way to hack into running FSI process in order to get access to autocomplete APIs. It is possible to use reflection in order to explore type system of a running process, but we need an entry point in the process - reference to some variable or static field. So I carefully studied FSI source code, but didn't find anything I could use for that purpose.  
I tried hard to find a way to find a reference to the class or variable of the running process given that I know that there in only one instance of that class exists in a process. It turned out impossible, or at least I didn't find any reasonable approach to do that. Another approach I investigated is hacking running process itself. There are ways to do that using CLR Profiler APIs, but that is a bit too extreme to be usable. After struggling with this for a while, I've got an idea. The fact that I didn't find any "entry" points into FSI process, doesn't mean that they don't exist. And if they exist, I should be able to see them in memory profiler. So I downloaded [CLR profiler](https://clrprofiler.codeplex.com/), and references view gave me the answer. System.AppDomain is a globally accessible by System.AppDomain.CurrentDomain, and I know that there is only one subscriber to \_unhandledException event. So this is our entry point.  
  

![](/assets/images/implementing-f-interactive-intellisense/image-1.png)

  
Here is how code for executing autocomplete API on a running FSI process end up looking:  
  

```fsharp
let getCompletionsFromFsiSession(prefix:String) : seq<String> = 
    try
        let fsiEvaluationSession = System.AppDomain.CurrentDomain 
                                |> getField "_unhandledException" 
                                |> getProperty "Target"
                                |> getField "callback"
                                |> getField "r"
                                |> getField "f"
                                |> getField "x"
 
        let fsiIntellisenseProvider = fsiEvaluationSession |> getField "fsiIntellisenseProvider"
        let istateRef = fsiEvaluationSession |> getField "istateRef"
 
        let getCompletionsFromFsiSession(prefix:String) = 
            fsiIntellisenseProvider |> invokeMethod "CompletionsForPartialLID" [|istateRef |> getProperty "contents"; prefix|] :?> seq<String>
 
        let changeLastLetterCase(prefix:String) = 
            if String.IsNullOrEmpty(prefix) then
                prefix
            else
                let lastChar = prefix.[prefix.Length - 1]
                let updatedLastChar = 
                    if Char.IsLower(lastChar) then
                        Char.ToUpper(lastChar)
                    else if Char.IsUpper(lastChar) then
                        Char.ToLower(lastChar)
                    else
                        lastChar
                prefix.Remove(prefix.Length - 1) + updatedLastChar.ToString()
 
        let changedLastLetterCasePrefix = changeLastLetterCase(prefix)
        let completions = getCompletionsFromFsiSession(prefix)
        let changedLastLetterCaseCompletions = if prefix = changedLastLetterCasePrefix then Seq.empty else getCompletionsFromFsiSession(changedLastLetterCasePrefix)
        
        completions
        |> Seq.append(changedLastLetterCaseCompletions) 
        |> Seq.distinct
        |> Seq.sort
    with 
        | _ -> Seq.empty
```

  
Extremely hacky, but not too complicated once you know what to do.  
  
Since this API doesn't tell what kind of completion it returns (variable, method, class, etc.), I merged its results with output of my home grown provider in order to be able to show icons.  
  
And the last piece was configuration, luckily Visual Studio has infrastructure to make it easy:  
<https://msdn.microsoft.com/en-us/library/bb166195.aspx>.  
  
So finally everything worked. Intellisense support is still far from perfect, but much better than nothing. I guess I'll try to integrate it with <https://github.com/fsharp/FsAutoComplete> later, we'll see.

As a result I've finally got decent Intellisense support for FSI, but it took 10-20 times of my original estimate to get there. Not sure that it was worth of time spent, but it was definitely interesting and unconventional project, so I had some fun while building it. Hope anyone will find it useful. Here is a link to GitHub project: <https://github.com/vlasenkoalexey/FSharp.Interactive.Intellisense>
