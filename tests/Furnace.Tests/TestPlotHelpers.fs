// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System
open System.IO
open NUnit.Framework
open Furnace
open Furnace.Util

[<TestFixture>]
type TestPlotHelpers() =

    [<Test>]
    member _.TestHelpersPrintValFloat32() =
        // Test printVal with Float32 values
        let result1 = printVal (3.14f :> scalar)
        let result2 = printVal (Single.NaN :> scalar)
        let result3 = printVal (Single.PositiveInfinity :> scalar)
        
        Assert.IsTrue(result1.Contains("3.14"))
        Assert.AreEqual("float('nan')", result2)
        Assert.AreEqual("float('inf')", result3)

    [<Test>]
    member _.TestHelpersPrintValFloat64() =
        // Test printVal with Float64 values
        let result1 = printVal (2.718 :> scalar)
        let result2 = printVal (Double.NaN :> scalar)
        let result3 = printVal (Double.PositiveInfinity :> scalar)
        
        Assert.IsTrue(result1.Contains("2.718"))
        Assert.AreEqual("float('nan')", result2)
        Assert.AreEqual("float('inf')", result3)

    [<Test>]
    member _.TestHelpersPrintValIntegers() =
        // Test printVal with various integer types
        let result1 = printVal (42 :> scalar)        // Int32
        let result2 = printVal (42L :> scalar)       // Int64
        let result3 = printVal (42s :> scalar)       // Int16
        let result4 = printVal (42uy :> scalar)      // Byte
        let result5 = printVal (42y :> scalar)       // SByte
        
        Assert.AreEqual("42", result1)
        Assert.AreEqual("42", result2)
        Assert.AreEqual("42", result3)
        Assert.AreEqual("42", result4)
        Assert.AreEqual("42", result5)

    [<Test>]
    member _.TestHelpersPrintValBoolean() =
        // Test printVal with Boolean values
        let resultTrue = printVal (true :> scalar)
        let resultFalse = printVal (false :> scalar)
        
        Assert.AreEqual("True", resultTrue)
        Assert.AreEqual("False", resultFalse)

    [<Test>]
    member _.TestHelpersToPythonBool() =
        // Test toPython with boolean values
        let resultTrue = toPython true
        let resultFalse = toPython false
        
        Assert.AreEqual("True", resultTrue)
        Assert.AreEqual("False", resultFalse)

    [<Test>]
    member _.TestHelpersToPythonScalarTensor() =
        // Test toPython with scalar tensor
        let t = FurnaceImage.scalar(42.0f)
        let result = toPython t
        
        Assert.IsTrue(result.Contains("42"))

    [<Test>]
    member _.TestHelpersToPython1DTensor() =
        // Test toPython with 1D tensor
        let t = FurnaceImage.tensor([1.0f; 2.0f; 3.0f])
        let result = toPython t
        
        // Should be in Python list format: [1.000000, 2.000000, 3.000000]
        Assert.IsTrue(result.StartsWith("["))
        Assert.IsTrue(result.EndsWith("]"))
        Assert.IsTrue(result.Contains("1."))
        Assert.IsTrue(result.Contains("2."))
        Assert.IsTrue(result.Contains("3."))

    [<Test>]
    member _.TestHelpersToPython2DTensor() =
        // Test toPython with 2D tensor
        let t = FurnaceImage.tensor([[1.0f; 2.0f]; [3.0f; 4.0f]])
        let result = toPython t
        
        // Should be nested list format: [[1., 2.], [3., 4.]]
        Assert.IsTrue(result.StartsWith("["))
        Assert.IsTrue(result.EndsWith("]"))
        // Should contain at least two opening brackets for nested structure
        let openBrackets = result.ToCharArray() |> Array.filter (fun c -> c = '[') |> Array.length
        Assert.GreaterOrEqual(openBrackets, 2)

    [<Test>]
    member _.TestHelpersToPythonOtherTypes() =
        // Test toPython with other types (should fall back to ToString)
        let result = toPython "hello world"
        Assert.AreEqual("hello world", result)
        
        let result2 = toPython 123
        Assert.AreEqual("123", result2)

    [<Test>]
    member _.TestHelpersRunScriptSuccess() =
        // Test runScript with successful execution (echo command)
        let tempDir = Path.GetTempPath()
        let lines = [| "echo 'test'" |]
        
        // This should not throw an exception
        Assert.DoesNotThrow(fun () -> runScript "echo" lines 1000)

    [<Test>]
    member _.TestHelpersRunScriptTimeout() =
        // Test runScript with timeout (should handle gracefully)
        let lines = [| "sleep 5" |]  // Command that takes longer than timeout
        
        // This should not throw an exception, just print warning
        Assert.DoesNotThrow(fun () -> runScript "sleep" lines 100)

    [<Test>]
    member _.TestHelpersRunScriptInvalidExecutable() =
        // Test runScript with invalid executable (should handle gracefully)
        let lines = [| "test" |]
        
        // This should not throw an exception, just print warning
        Assert.DoesNotThrow(fun () -> runScript "nonexistent_executable_12345" lines 1000)