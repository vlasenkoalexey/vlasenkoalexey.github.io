---
layout: post
title: "Monads explanation for C#/Java developers"
date: 2016-11-30
categories: [Programming, Functional Programming]
tags: [C#, F#, Haskell, Java, Monad]
original_url: "https://alekseyv-dev.blogspot.com/2016/11/monads-explanation-for-cjava-developers.html"
---

It so happened that October turned out to be a month of Haskell for me. Accidentally I found a new [Haskell MOOC](https://www.futurelearn.com/courses/functional-programming-haskell) by University of Glasgow. I got curious mostly because University of Glasgow is known as the birthplace of most commonly used Haskell compiler.  
  
Later during same month I got into Haskell 101 and 201 classes in Google, that turned out to be short but extremely good. One of things I liked the most about our internal classes is the approach for describing Monads. I was a bit anxious to watch this topic in Glasgow MOOC, but it was disappointing. They did very good for explaining basic concepts, but didn't even record a video dedicated to the most dreaded Haskell topic. Monads are conceptually simple ... for Haskell or other functional language developers. But they definitely aren't for mainstream developers.  
  
My first encounter with Monad happened when I studied F# around 5 years ago. Everything went well, until I got to the topic of computational expressions. I spent some time trying to figure out this concept without much success. I figured that they are similar to mysterious Monads in Haskell. At that time I just learned how to use async which is the most common computation expression use case in F# and moved on.  
  
Second encounter happened in [FP101x](https://courses.edx.org/courses/DelftX/FP101x/3T2014/info) class. I took it mostly to figure out Monad concept, but again it didn't get enough attention here. The common joke about Monad is that once people master the concept, they become unable to explain it to others. This is only a half joke I guess, because most Monad tutorials are written in Haskell. The other reason is that concept is pretty basic and intuitive ... for developers who deal with functional programming languages. Once concept is mastered such developer will try to explain it using examples from functional world. I have to admit that it took quite a while before I got some grasp on Monads. This wasn't easy, I spent hours watching videos and reading tutorials. There are plenty of those available, but I didn't find any that would work perfectly for me. From my personal perspective most tutorials have following flaws:  

- Most of tutorials are written in Haskell or Scala. If you don't know either of them, it is much harder to digest samples. This is especially true for Haskell which is using a lot of cryptic functions like >>=. So you either have to learn Haskell first (which is the right way), or move on.
- Don't account for audience background. I guess people who are interested in Monads can be divided into 2 groups. Those who are learning Haskell or any other FP language and got to the topic of Monads, or professional developers who are just interested in a Monad concept. In both cases Haskell is not going to be the first programming language. Therefore it is reasonable to assume familiarity with either C# or Java.
- Some tutorials are explained with theory first in mind. I think I got enlightenment at the moment when I decided to figure out how each specific Monad works, and get to the general concept after that.
- Some tutorials are very long and consist of series of posts. Regardless of commitment level, it is quite hard to get to the core when reading long quite abstract text.
- Not all tutorials explain why Monads matter in a first place.

Ideally I wanted to find a tutorial which uses simple terms with intuitive examples in C#/Java and short enough to be consumable in one attempt. I didn't find anything like that, so here is my attempt to create one. Basically this is the tutorial I'd wanted to have when I was struggling to understand concept few years ago. Hope my explanation will make sense primarily because I'm not dealing with functional programming in my day to day work and clearly remember problems I faced when trying to figure out Monads on my own.

  
Why Monads matter  
  
I think the first and most obvious question is why Monads matter at all? Indeed Monads are not necessary for imperative programming languages, but still can be useful. You can think about Monad as a design pattern in OO world. By definition design pattern allows to solve problems in a consistent manner. You can write code without using design patterns, but it is getting cleaner and more reusable if you apply them properly.  
Same statement is applicable to Monads even to higher extent. In fact Monad pattern become so ubiquitous in functional languages, that most functional programming language compilers got syntactic sugar support for Monads.  
  
Second question is why all Monads tutorials are written in Haskell which is pretty hard to read if you normally write code in C derived language. This one is pretty simple, Monads matter the most for pure programming languages which do not support mutable state (you can't change variable once you set it). Haskell is the only popular pure functional programming language.  
  
So finally what Monad is? Conceptually this is a design pattern to wrap some value into the class and expose certain operations to manipulate it. From OO perspective we can think about it as a wrapper class which implements an interface with constructor and a method to change wrapped value. By convention constructor function is called **return** or sometimes **unit**, and manipulation function is called **bind.**  
  
To get an intuition you can think about Monad as a class which supports following interface:  
  

```
// Pseudo code, for demonstation purposes only.
public interface IMonad<A>
{
    // Monadic constructor (psuedo code since interfaces to not support constructors).
    IMonad<A> Return<A>(A value); 
    IMonad<B> Bind<A, B>(Func<A, IMonad<B>> func);
}
```

  
I guess it is still not clear why we need it, and why it is useful. There are few canonical examples.  
  
Let's say that we have value of type A:  
  
A -> value  
  
If we want to implement an operation which might fail we can return null as we usually do in C#/Java, or wrap it in Nullable/Optional. In functional world there is a convention to call this class Maybe. With Maybe (if used correctly) we won't have to care about null exceptions. Maybe can either store value - Just<int>(1) or be empty - Nothing.  
  
Maybe A -> optional value  
  
If we want to use multiple values of type A, we wrap it in the List (think IEnumerable/Iterable)  
  
List A -> repeated value  
  
If we need to deal with IO in pure language, or async IO (Task/Future), we can also wrap it in a Monadic class  
  
IO A -> impure value, or value from async computation  
  
Wrapping value in Monadic class is trivial, we just need to call a constructor - return(value) which is translated to following:  
  
Maybe A -> new Just<A>(value);  
List A -> new List<A>(value);  
IO A -> new IO<A>(value);  
  
Unwrapping value is not trivial and in some cases not possible.  
  
Maybe A -> can't unwrap Nothing  
List A -> can't unwrap empty list  
IO A -> can't unwrap IO/async operation.   
  
In C# terms once you declare async method, all caller methods should also be async (you can get out of async code by calling Result method on Task after it completed, but this is not recommended).  
  
Since we can't unwrap the value, we can let Monad to be responsible for manipulating it. That is what **bind** method is used for. Let's look at few examples in C#. **bind** method takes a function which takes a value of same type and returns a Monad instance (think wrapper class) of another type (which might be same type).  
  
Maybe Monad  
  

```
public abstract class Maybe<A>
{
    public static Maybe<A> Return(A value) //emulating Monadic constructor
    {
        return new Just<A>(value);
    }
 
    public Maybe<B> Bind<A, B>(Func<A, Maybe<B>> func)
    {
        var justa = this as Just<A>;
        return justa == null ?
            new Nothing<B>() :
            func(justa.Value);
    }
}
 
public class Nothing<A> : Maybe<A>
{
    public override string ToString()
    {
        return "Nothing";
    }
}
 
public class Just<A> : Maybe<A>
{
    public A Value { get; private set; }
    internal Just(A value) // Monadic constructor
    {
        Value = value;
    }
    public override string ToString()
    {
        return Value.ToString();
    }
}
```

  
Note  that base classed is used here to simulate a [discriminated union](https://docs.microsoft.com/en-us/dotnet/articles/fsharp/language-reference/discriminated-unions) data structure which is only available in functional languages.  Use case is pretty clear.  
Normally if we have class hierarchy which is few level deep, we have to check each object for null:  
  

```
public class User
{
    public string UserName { get; set; }
    public Order UserOrder { get; set; }
}
 
public class Order
{
    public OrderTracking Tracking { get; set; }
}
 
public class OrderTracking
{
    public string TrackingNumber { get; set; }
    public string Carrier { get; set; }
}
 
public string GetTrackingNumber(User user)
{
    if (user != null)
    {
        var order = user.UserOrder;
        if (order != null)
        {
            var tracking = order.Tracking;
            if (tracking != null)
            {
                return tracking.TrackingNumber;
            }
        }
    }
 
    return null;
}
```

  

This is a mess, and in order to make it nicer C# 6.0 got a null-conditional operator (?.). People argued that this operator violates [Law of Demeter](http://softwareengineering.stackexchange.com/questions/299286/does-c-6-0s-new-null-conditional-operator-go-against-the-law-of-demeter), and I guess this is one of the reasons why it was added so late.  
  

```
public string GetTrackingNumberShort(User user)
{
    return user?.UserOrder?.Tracking?.TrackingNumber;
}
```

  
Looks much better, but it doesn't feel that problem is solved mainly because you never know when to expect null and therefore in a spirit of defensive programming have to add null checks everywhere.  
  
Here is how this code looks like with Maybe:  
  

```
public class User
{
    public string UserName { get; set; }
    public Maybe<Order> UserOrder { get; set; }
}
 
public class Order
{
    public Maybe<OrderTracking> Tracking { get; set; }
}
 
public class OrderTracking
{
    public Maybe<string> TrackingNumber { get; set; }
    public string Carrier { get; set; }
}
 
public Maybe<string> GetTrackingNumber(User user)
{
    return user.UserOrder
        .Bind<Order, OrderTracking>((order) => order.Tracking)
        .Bind<OrderTracking, String>((tracking) => tracking.TrackingNumber);
}
```

  
Bind method will only pass through value further if it has value, otherwise all Bind chain will return Nothing. In a C# this looks like overkill, but this is a powerful concept and it is worth to explore it further.  
  
List Monad  
  
The List Monad monad implementation is not as intuitive as the one for Maybe. The thing is generic C# List is conceptually a Monad, but instead of Bind method it has SelectMany exposed by Linq. List Monad is used to wrap multiple values of same type and handle use case when List is empty. It is also worth mentioning that functional list is mostly similar to IEnumerable/Iterative in C#/Java.  
  
In order to build intuition about List Monad, here is C# implementation with aggregated IEnumerable:  
  

```
public class ListMonad<A>
{
    IEnumerable<A> elements;
   
    public ListMonad<B> Bind<B>(Func<A, ListMonad<B>> func)
    {
        List<B> results = new List<B>();
 
        foreach (A element in elements)
        {
            foreach (B result in func(element).elements)
            {
                results.Add(result);
            }
        }
 
        return new ListMonad<B>(results);
    }
 
    // Alternative implementation
    public ListMonad<B> BindAlt<B>(Func<A, ListMonad<B>> func)
    {
        // SelectMany is also known as flatMap
        return new ListMonad<B>(elements.SelectMany(element => func(element).elements));
    }
 
    public static ListMonad<A> Return(A element)
    {
        return new ListMonad<A>(new List<A> { element });
    }
 
    private ListMonad(IEnumerable<A> elements)
    {
        this.elements = elements;
    }
}
```

  
And most naive implementation using extension method:  
  

```
public static class ListMonadAsExtensions
{
    public static IEnumerable<B> Bind<A, B>(this IEnumerable<A> value, 
        Func<A, IEnumerable<B>> func)
    {
        return value.SelectMany(element => func(element));
    }
 
    public static IEnumerable<A> Return<A>(A element)
    {
        return new List<A> { element };
    }
}
```

  
Usage demonstration:  
  

```
public class ListMonadSample
{
    public class User
    {
        public string Name { get; set; }
 
        public ListMonad<User> Children { get; set; }
    }
 
    public ListMonad<string> GetGrandChildrenNamesLinq(User user)
    {
        return user.Children
            .Bind(child => child.Children)
            .Bind(grandChild => ListMonad<String>.Return(grandChild.Name));
    }
}
```

  
It isn't as useful as Maybe because foreach doesn't fail on empty list but at least this should provide a proof that same very abstract concept can be applied to very different types.  
  
IO/Task Monad  
  
For C#/Java developer I/O is not a big deal, this is just a set of APIs to provide methods for interacting with outside world. It is very different with pure languages which do not allow to change values. That is why all I/O in Haskell is implemented using Monad. Haskell is a lazy language (like C# Linq) and it postpones function execution until results are necessary. Since all functions do not have side effects (they won't modify any state or interact with IO), functions can be composed into other functions. This all works nice until the moment you need to interact with IO, once you have something to read for example you need to pause function execution and wait for input.  
  
In this respect I think async Task is a great analogy. Result of async Task is only going to be available once Task execution is complete. In fact [Task is conceptually a Monad](http://stackoverflow.com/questions/15611972/how-does-c-sharp-async-await-relates-to-more-general-constructs-e-g-f-workflo), except it doesn't expose necessary methods.  
  
Here is a IO/Task Monad implementation:  
  

```
public static class TaskMonadExtensions
{
    public static Task<T> Return<T>(this T value)
    {
        return Task.FromResult(value);
    }
 
    public static async Task<TNewResult> Bind<TResult, TNewResult>(this Task<TResult> task, 
        Func<TResult, Task<TNewResult>> continuation)
    {
        return await continuation(await task);
    }
 
    public static Task<TNewResult> BindAlt<TResult, TNewResult>(this Task<TResult> task, 
        Func<TResult, Task<TNewResult>> continuation)
    {
        // For demo purposes only, this won't handle exceptions and cancellations properly.
        return task.ContinueWith(t => continuation(t.Result)).Unwrap();
    }
}
```

  
See this link for full implementation of [task chaining](https://blogs.msdn.microsoft.com/pfxteam/2010/11/21/processing-sequences-of-asynchronous-operations-with-tasks/) using ContinueWith.  
  
Usage sample is rather trivial:  
  

```
public static async void TaskMonadSampleWithAsync()
{
    int userId = 1;
    var userCode = await GetUserNameById(userId).Bind(name => GetUserCode(name));
    Console.WriteLine("TaskMonadSampleWithAsync, user code: {0}", userCode);
}
 
public static void TaskMonadSampleWithoutAsync()
{
    int userId = 1;
    Task<int> task = GetUserNameById(userId).BindAlt(name => GetUserCode(name));
    task.Wait();
    Console.WriteLine("TaskMonadSampleWithoutAsync, user code: {0}", task.Result);
}
 
public static async void TaskSampleWithAsync()
{
    int userId = 1;
    string userName = await GetUserNameById(userId);
    int userCode = await GetUserCode(userName);
    Console.WriteLine("TaskSampleWithAsync, user code: {0}", userCode);
}
 
public static void TaskSampleWithoutAsync()
{
    int userId = 1;
    Task<int> task = GetUserNameById(userId).ContinueWith(t => GetUserCode(t.Result)).Unwrap();
    task.Wait();
    Console.WriteLine("TaskSampleWithoutAsync, user code: {0}", task.Result);
}
 
 
static Task<String> GetUserNameById(int userId)
{
    return Task<String>.Factory.StartNew(
        (obj) =>
        {
            // Simulate a slow operation
            Thread.Sleep(100);
            Console.WriteLine("- GetUserNameById: Thread={0}, userId={1}", 
                Thread.CurrentThread.ManagedThreadId, obj);
            return "aleksey";
        },
        userId);
}
 
static Task<int> GetUserCode(string name)
{
    return Task<int>.Factory.StartNew(
        (obj) =>
        {
            // Simulate a slow operation
            Thread.Sleep(100);
            Console.WriteLine("- GetUserCode: Thread={0}, name={1}", 
                Thread.CurrentThread.ManagedThreadId, obj);
            return name.GetHashCode();
        },
        name);
}
```

  
In order to simplify asynchronous programming C# 5.0 introduced async/await keywords. People argued that it would be better to [generalize it as a Monad](http://stackoverflow.com/questions/15611972/how-does-c-sharp-async-await-relates-to-more-general-constructs-e-g-f-workflo), but Microsoft choose to keep things simple. C# always has been practical first language.  
  
Putting all pieces together  
  
I think at this moment it should be clear that concept of wrapping value and exposing consistent method for transforming it can be applied to different areas. But it is still not very clear how we can benefit from it. Let's see some more a less practical example. Let say that we have a chain of asynchronous API calls where any method can return Nothing/null.  
  

```
public Task<Maybe<String>> GetData() ...
public Task<Maybe<String>> GetMoreData(Maybe<String> data) ... 
public Task<Maybe<String>> SendProcessedResults(Maybe<String> data) ...
public Maybe<String> Validate(String data) ...
public Maybe<String> Sanitize(String data) ...
public Maybe<String> Process(String data) ...
 
public Task<Maybe<String>> RunWorkflow()
{
    return GetData().Bind(data =>
    {
        Maybe<String> processedData = data
            .Bind(raw => Validate(raw)
                .Bind(validated => Sanitize(validated)
                    .Bind(sanitized => Process(sanitized))));
 
        return GetMoreData(processedData).Bind(extraData =>
        {
            Maybe<String> processedExtraData = extraData
                .Bind(raw => Validate(raw)
                    .Bind(validated => Sanitize(validated)
                        .Bind(sanitized => Process(sanitized))));
 
            return SendProcessedResults(processedExtraData);
        });
    });
}

// Another version
public Task<Maybe<String>> RunWorkflowFlat()
{
    return GetData().Bind(data =>
    {
        Maybe<String> processedData = data
            .Bind(raw => Validate(raw))
            .Bind(validated => Sanitize(validated))
            .Bind(sanitized => Process(sanitized));
 
        return GetMoreData(processedData);
    }).Bind(extraData =>
    {
        Maybe<String> processedData = extraData
            .Bind(raw => Validate(raw))
            .Bind(validated => Sanitize(validated))
            .Bind(sanitized => Process(sanitized));
 
        return SendProcessedResults(processedData);
    });
}
```

  
When executed this sample returns

"send\_result: data validated sanitized processed more validated sanitized processed"

wrapped into Maybe class.  
  
As you see here it is possible to mix Monad **Bind** operations as long as you keep them separate. I mean that you won't be able to pass string to GetMoreData by using Maybe.Bind without deconstructing Maybe Monad.  
  
Each lambda expression here is executed only if some criteria is met - in case of Maybe value should not be empty, and in case of async Task previous operation should be completed.  
  
Here we see that same approach is generalized to work for different Monad types. Can we somehow take advantage of this? Yes we can, but unfortunately not in C# or Java. Functional languages has special syntactic sugar for Monads which makes their application effortless.  
  
Monads in Haskell  
  
Now it should be clear how to implement most common Monads in C#, and how to use them.  
But it is not quite obvious why this concept is helpful and so popular. Let's look how to use Monads in Haskell.  
  
Here is a simplified version of Maybe Monad (trailing underscore is used to avoid conflict with built-in Maybe implementation).  
  

```
-- maybe is a discriminated union that contains either wrapped 'a' value or Nothing_.
data Maybe_ a = Nothing_ | Just_ a 
  deriving (Show, Eq, Ord) -- Hasell shoud provide default implementations for common methods.

instance Monad Maybe_ where
   return x = Just_ x
   (>>=) maybe f = -- (>>==) stands for bind in Haskell
       case maybe of -- doing pattern matching on maybe
           Just_ x -> f x -- if maybe has value, apply function f to the wrapped value
           Nothing_ -> Nothing_ -- otherwise return Nothing_
```

  
There is nothing special here except for **(>>=)** function which is used instead of **bind** in Haskell.  
Symbolic functions (like + or \*) in Haskell are infix by default. It means that it is used between arguments like 4 + 5. In order use it like prefix function, it is necessary to wrap it in brackets like  
(+) 4 5. Therefore by default >>= is infix operator used in form of:  
  
monad >>= bind function (f in Maybe case)  
  
>>= for Maybe usually defined like:  
  

```
instance Monad Maybe_ where
   Nothing_  >>= _ = Nothing_ -- >>== stands for bind
   (Just_ x) >>= f =  sf x -- it is declared here as infix function
   return x = Just_ x
```

  
This definition is equivalent, but very unusual if you see it for the first time.  
  
Now as we familiar with syntax, let's see how same Monad sample will look like in Haskell:  
  

```
getData :: IO (Maybe String) -- type declaration
-- return here stands for wrapping Maybe String into IO Monad
getData = return (Just "data") 
 
getMoreData :: Maybe String -> IO (Maybe String)
getMoreData dataOpt = 
    let moreDataOpt = do 
        d <- dataOpt -- unwraping maybe Monad
        -- ++ operator is used for string concatination
        return (d ++ " more") -- return data out of do block
    in return(moreDataOpt) -- wrapping moreDataOpt in IO

sendProcessedResults :: Maybe String -> IO (Maybe String)
sendProcessedResults dataOpt = 
    let sendResultOpt = do 
        d <- dataOpt
        return ("send_result: " ++ d) -- return data out of do block
    in return(sendResultOpt) -- wrapping moreDataOpt in IO

-- Haskell doesn't use brackets for function declaraions
-- It is not necessary to specify function type if compiler can deduct it.
validate d = Just(d ++ " validated") 
sanitize d = Just(d ++ " sanitized") 
process d = Just(d ++ " processed")

-- function without parametes
runWorflow = 
    getData >>= \rawOpt -> -- \rawOpt -> ... is a lambda function with rawOpt parameter
        let processedData = (
                rawOpt >>= \raw -> -- \raw -> ... is a lambda function with raw parameter
                    validate raw >>= \validated ->
                        sanitize validated >>= \sanitized ->
                            process sanitized)
        in getMoreData processedData >>= \moreDataOpt -> 
            let processedMoreResult = (
                    moreDataOpt >>= \raw -> 
                        validate raw >>= \validated ->
                            sanitize validated >>= \sanitized ->
                                process sanitized)
            in sendProcessedResults processedMoreResult
```

  
runWorkflow in this snippet is barely readable but conceptually similar to the one in C#.  
In order to make such code readable, Haskell introduced [do notation](https://en.wikibooks.org/wiki/Haskell/do_notation):  
  

```
 -- function without parametes
runWorflow = 
    do -- IO block
        rawOpt <- getData -- unwraping IO monad
        
        let processedResult = do  -- nested maybe block
            raw <- rawOpt
            validated <- validate raw
            sanitized <- sanitize validated
            processed <- process sanitized
            return processed -- getting out of maybe do block
        
        moreDataOpt <- getMoreData processedResult
 
        let processedMoreResult = do -- nested maybe block
            raw <- moreDataOpt
            validated <- validate raw
            sanitized <- sanitize validated
            processed <- process sanitized
            return processed -- getting out of maybe do block

        results <- sendProcessedResults processedMoreResult
        return results -- getting out of IO block
```

  
This code might look like imperative, but in reality it is same as the snippet above. <- operator in do block "binds" value on the left side to the function passed to the **>>=/bind**. **return** statement here is used to get result out of do block.  
  
This pattern is very popular in Haskell mostly because it is a pure language (doesn't allow state changes) and all I/O is implemented through Monad. If you can't change variable and need to read something from console or other external API, only way to do it is by declaring a function where argument is a value returned by that API. That is a function passed to the **>>=/bind**. Once computation is wrapped in impure IO Monad, it is not possible to get value out of it. Only way to work with such value is by functions provided to **>>=/bind** method.  
  
Computational expressions (aka Monads) in F#  
  
Let's see how same sample looks in F#. The thing is F# has optional type which is similar to Nullable in C#, but it doesn't support monadic operations out of box. Not a problem, it is a great exercise to build it yourself:  
  

```
module Builders = 
    type MaybeBuilder() =
 
        member this.Bind(x, f) = 
            match x with
            | None -> None
            | Some a -> f a
 
        member this.Return(x) = 
            Some x
 
        member this.ReturnFrom x = x
   
    let maybe = new MaybeBuilder()
```

  
You might not understand F# syntax, but probably it is easy to notice that this is somewhat similar to Maybe implementation in C# and Haskell. In F# Monads are called computational expressions (they are similar but not quite same), and there is a concept of "builders" in order to work with computational expressions. I don't want to get into details, here is a great [article](https://fsharpforfunandprofit.com/posts/computation-expressions-intro/) about computational expressions.  
And here is same combined Monad sample implemented in F#:  
  

```
open System
open Builders
 
let getData() = async { return Some("data") }
 
let getMoreData(data : Option<String>) = async {
        return maybe.Bind(data, fun data -> Some(data + " more"))
    }
 
let sendProcessedResults(data : Option<String>) = 
    async {
        return maybe.Bind(data, fun data -> Some("send_result: " + data))
    }
 
let validate(data:String) = Some(data + " validated")
let sanitize(data:String) = Some(data + " sanitized")
let processData(data:String) = Some(data + " processed")
 
let runWorflow() = // return type is Async<String<Option>>
    async {
        let! rawOpt = getData() // rawOpt is of type Option<String>
        let processed = maybe { // processed is of type Option<String>
            let! raw = rawOpt 
            let! validated = validate(raw)
            let! sanitized = sanitize(validated)
            return! processData(sanitized)
        }
 
        let! moreDataOpt = getMoreData(processed) // moreData is of type Option<String>
 
        let processedMore = maybe { // processedMore is of type Option<String>
            let! raw = moreDataOpt
            let! validated = validate(raw)
            let! sanitized = sanitize(validated)
            return! processData(sanitized)
        }
 
        return! sendProcessedResults(processedMore)
    }
```

  
There are few interesting things here, particularly **async** and **maybe** blocks, and **let!**, **return!** statements. **async** and **maybe** blocks are similar to do notation in Haskell and used to mark scope where bang operations like **let!** and **return!** can be used. **let!** operation invokes Builder's Bind method. **let!** raw = rawOpt is equivalent to rawOpt.Bind(raw => (all code below)). **return!** invokes ReturnFrom method which just returns value.  
Basic rule for bang operators is that right side of expression is wrapped (Monadic value), and unwrapped value on the left side. Normal let operations are also permitted in the Builder blocks. Builders and bang operators are F# compiler syntactic sugar to simplify Monadic operations. Isn't this beautiful?  
  
Conclusion  
  
It is also important to mention that unlike C#, F# doesn't have special syntax for async operations. In F# async is implemented as a computational expression similarly to maybe implementation above. I think this is one of the reasons why F# got async support long before C# did. And regarding maybe, in most of cases it can replace ?? and ?. C# operators. I think conclusion to this is clear, the concept of Monad is an abstract but a very powerful one. Languages that support Monads can be 'extended' without adding new keywords and operators.  
  
As I mentioned before Monad as a concept still can be useful for C#/Java, C# [Sprache](https://github.com/sprache/Sprache) monadic parser combinator framework is a great example.  
  
Still, there is more...  
For simplicity sake I skipped few very important points:  

- Monad implementations should support [Monadic laws](https://wiki.haskell.org/Monad_laws)
- Monad is [generalization](http://learnyouahaskell.com/functors-applicative-functors-and-monoids) on top of Applicative which is generalization on top of Functor
- Monadic classes also support few other [operations](http://learnyouahaskell.com/a-fistful-of-monads)

  
I don't want to get into those topics here, but I hope that this article helps with building intuition about Monads and their application, and makes exploring those more advanced topics easier.  
  
All code samples can be found on [GitHub](https://github.com/vlasenkoalexey/CSharpMonadSamples).  
  
References and Credits  
Here are links to the resources which helped me to understand Monads. Comments reflect my personal opinion.  
  
[Yet Another Monad Tutorial](http://mvanier.livejournal.com/3917.html) - I think this is the very best Monads tutorial. This is the article series which helped me the most. Tutorial is using Haskell and rather deep and long, not an easy read. As I mentioned before most important insight I got here is that it is easier to learn how each Monad works first, and see how it generalizes to the common concept.

 [Monads and Superheroes](https://www.youtube.com/watch?v=MlZCiiKGbb0) - probably the best video presentation on Monads. I feel that examples are too detailed and not easy to digest.  

[Monads by Eric Lippert](https://ericlippert.com/category/monads/) - very long and detailed explanation for C# developers. Not an easy read.

[Monads in C#](http://mikehadlow.blogspot.com/2011/01/monads-in-c1-introduction.html) - rather short series with great ideas, but as for me not well structured.

[Monads, a design pattern](https://www.stephanboyer.com/post/9/monads-part-1-a-design-pattern) - short explanation in Python.

[Monads in Scala](http://scabl.blogspot.com/2013/02/monads-in-scala-1.html) - short, but good explanation.

[Functors, Applicatives, And Monads In Pictures](http://adit.io/posts/2013-04-17-functors,_applicatives,_and_monads_in_pictures.html) - great tutorial in pictures, but I didn't get it before I got some context.  

[Learn You a Haskell for Great Good](http://learnyouahaskell.com/functors-applicative-functors-and-monoids) - free Haskell book, has few chapters dedicated to Monads. I didn't get the concept the first time I read it. Makes much more sense to me now.
