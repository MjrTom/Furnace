// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System
open Furnace
open Furnace.Backends.Reference
open NUnit.Framework
open Tests.TestUtils

[<TestFixture>]
type TestReferenceUtils () =
    
    [<Test>]
    member _.TestGetTypedValuesFloat32() =
        // Test GetTypedValues extension method on float32 tensors
        let values = [|1.0f; 2.0f; 3.0f; 4.0f|]
        let shape = Shape.create [|2; 2|]
        let tensor = RawTensorFloat32(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<float32>()
        Assert.AreEqual(values, extracted)
        Assert.AreEqual(4, extracted.Length)
        Assert.AreEqual(1.0f, extracted.[0])
        Assert.AreEqual(4.0f, extracted.[3])
    
    [<Test>]
    member _.TestGetTypedValuesInt32() =
        // Test GetTypedValues extension method on int32 tensors
        let values = [|10; 20; 30; 40|]
        let shape = Shape.create [|2; 2|]
        let tensor = RawTensorInt32(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<int32>()
        Assert.AreEqual(values, extracted)
        Assert.AreEqual(4, extracted.Length)
        Assert.AreEqual(10, extracted.[0])
        Assert.AreEqual(40, extracted.[3])
    
    [<Test>]
    member _.TestGetTypedValuesFloat64() =
        // Test GetTypedValues extension method on float64 tensors
        let values = [|1.5; 2.5; 3.5; 4.5|]
        let shape = Shape.create [|4|]
        let tensor = RawTensorFloat64(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<float64>()
        Assert.AreEqual(values, extracted)
        Assert.AreEqual(4, extracted.Length)
        Assert.AreEqual(1.5, extracted.[0])
        Assert.AreEqual(4.5, extracted.[3])
    
    [<Test>]
    member _.TestGetTypedValuesBool() =
        // Test GetTypedValues extension method on bool tensors
        let values = [|true; false; true; false|]
        let shape = Shape.create [|2; 2|]
        let tensor = RawTensorBool(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<bool>()
        Assert.AreEqual(values, extracted)
        Assert.AreEqual(4, extracted.Length)
        Assert.AreEqual(true, extracted.[0])
        Assert.AreEqual(false, extracted.[3])
    
    [<Test>]
    member _.TestGetTypedValuesByte() =
        // Test GetTypedValues extension method on byte tensors
        let values = [|255uy; 128uy; 64uy; 0uy|]
        let shape = Shape.create [|4|]
        let tensor = RawTensorByte(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<byte>()
        Assert.AreEqual(values, extracted)
        Assert.AreEqual(4, extracted.Length)
        Assert.AreEqual(255uy, extracted.[0])
        Assert.AreEqual(0uy, extracted.[3])

    [<Test>]
    member _.TestGetTypedValuesInt8() =
        // Test GetTypedValues extension method on int8 tensors
        let values = [|127y; -128y; 0y; 42y|]
        let shape = Shape.create [|2; 2|]
        let tensor = RawTensorInt8(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<int8>()
        Assert.AreEqual(values, extracted)
        Assert.AreEqual(4, extracted.Length)
        Assert.AreEqual(127y, extracted.[0])
        Assert.AreEqual(42y, extracted.[3])

    [<Test>]
    member _.TestGetTypedValuesInt16() =
        // Test GetTypedValues extension method on int16 tensors
        let values = [|1000s; -1000s; 0s; 32000s|]
        let shape = Shape.create [|4|]
        let tensor = RawTensorInt16(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<int16>()
        Assert.AreEqual(values, extracted)
        Assert.AreEqual(4, extracted.Length)
        Assert.AreEqual(1000s, extracted.[0])
        Assert.AreEqual(32000s, extracted.[3])

    [<Test>]
    member _.TestGetTypedValuesInt64() =
        // Test GetTypedValues extension method on int64 tensors
        let values = [|1000000L; -1000000L; 0L; 9223372036854775807L|]
        let shape = Shape.create [|2; 2|]
        let tensor = RawTensorInt64(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<int64>()
        Assert.AreEqual(values, extracted)
        Assert.AreEqual(4, extracted.Length)
        Assert.AreEqual(1000000L, extracted.[0])
        Assert.AreEqual(9223372036854775807L, extracted.[3])

    [<Test>]
    member _.TestGetTypedValuesWrongType() =
        // Test that casting to wrong type throws exception
        let values = [|1.0f; 2.0f; 3.0f; 4.0f|]
        let shape = Shape.create [|2; 2|]
        let tensor = RawTensorFloat32(values, shape, Device.CPU) :> RawTensor
        
        // This should throw when trying to cast float32 tensor to int32 values
        isException (fun () -> tensor.GetTypedValues<int32>())

    [<Test>]
    member _.TestGetTypedValuesScalar() =
        // Test GetTypedValues extension method on scalar tensors
        let values = [|42.0f|]
        let shape = Shape.create [||]  // scalar shape
        let tensor = RawTensorFloat32(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<float32>()
        Assert.AreEqual(1, extracted.Length)
        Assert.AreEqual(42.0f, extracted.[0])
        
    [<Test>]
    member _.TestGetTypedValuesEmpty() =
        // Test GetTypedValues extension method on empty tensors
        let values = [||]
        let shape = Shape.create [|0|]  // empty tensor
        let tensor = RawTensorFloat32(values, shape, Device.CPU) :> RawTensor
        
        let extracted = tensor.GetTypedValues<float32>()
        Assert.AreEqual(0, extracted.Length)