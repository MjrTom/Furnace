// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open Furnace
open Furnace.Backends.Torch

[<TestFixture>]
module TestTorchExtensions =

    [<Test>]
    let TestTorchExtensionsFromTorchTensor() =
        use config = FurnaceImage.useConfig(backend=Backend.Torch, dtype=Dtype.Float32)
        
        // Create a simple tensor with known values
        let values = [| 1.0f; 2.0f; 3.0f; 4.0f |]
        let originalTensor = FurnaceImage.tensor(values)
        
        // Convert to torch tensor and back
        let torchTensor = originalTensor.toTorch()
        let reconstructedTensor = FurnaceImage.fromTorch(torchTensor)
        
        // Verify the reconstructed tensor has the same values
        Assert.CheckEqual(originalTensor.shape, reconstructedTensor.shape)
        Assert.CheckEqual(originalTensor.dtype, reconstructedTensor.dtype)
        Assert.CheckEqual(originalTensor.backend, reconstructedTensor.backend)
        
        // Check values are preserved
        let originalArray = originalTensor.toArray() :?> float32[]
        let reconstructedArray = reconstructedTensor.toArray() :?> float32[]
        
        Assert.CheckEqual(originalArray.Length, reconstructedArray.Length)
        for i in 0 .. originalArray.Length - 1 do
            Assert.AreEqual(float originalArray[i], float reconstructedArray[i], 1e-6)

    [<Test>]
    let TestTorchExtensionsFromTorchTensor2D() =
        use config = FurnaceImage.useConfig(backend=Backend.Torch, dtype=Dtype.Float32)
        
        // Create a 2D tensor
        let values = [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |]
        let originalTensor = FurnaceImage.tensor(values)
        
        // Convert to torch tensor and back
        let torchTensor = originalTensor.toTorch()
        let reconstructedTensor = FurnaceImage.fromTorch(torchTensor)
        
        // Verify shape and properties
        Assert.CheckEqual([| 2; 2 |], reconstructedTensor.shape)
        Assert.CheckEqual(Dtype.Float32, reconstructedTensor.dtype)
        Assert.CheckEqual(Backend.Torch, reconstructedTensor.backend)
        
        // Check values (2D tensor returns float32[,])
        let originalArray = originalTensor.toArray() :?> float32[,]
        let reconstructedArray = reconstructedTensor.toArray() :?> float32[,]
        
        Assert.CheckEqual(originalArray.GetLength(0), reconstructedArray.GetLength(0))
        Assert.CheckEqual(originalArray.GetLength(1), reconstructedArray.GetLength(1))
        Assert.AreEqual(1.0, float reconstructedArray[0,0], 1e-6)
        Assert.AreEqual(2.0, float reconstructedArray[0,1], 1e-6)
        Assert.AreEqual(3.0, float reconstructedArray[1,0], 1e-6)
        Assert.AreEqual(4.0, float reconstructedArray[1,1], 1e-6)

    [<Test>]
    let TestTorchExtensionsFromTorchScalar() =
        use config = FurnaceImage.useConfig(backend=Backend.Torch, dtype=Dtype.Float32)
        
        // Create a scalar tensor
        let originalScalar = FurnaceImage.scalar(42.0f)
        
        // Convert to torch tensor and back
        let torchTensor = originalScalar.toTorch()
        let reconstructedScalar = FurnaceImage.fromTorch(torchTensor)
        
        // Verify scalar properties
        Assert.CheckEqual([||], reconstructedScalar.shape)  // Scalar has empty shape
        Assert.CheckEqual(Dtype.Float32, reconstructedScalar.dtype)
        Assert.CheckEqual(Backend.Torch, reconstructedScalar.backend)
        
        // Check the scalar value
        let originalValue = originalScalar.toScalar() :?> float32
        let reconstructedValue = reconstructedScalar.toScalar() :?> float32
        Assert.AreEqual(42.0, float reconstructedValue, 1e-6)
        Assert.AreEqual(float originalValue, float reconstructedValue, 1e-6)

    [<Test>]
    let TestTorchExtensionsToTorchWithDifferentDtypes() =
        use config = FurnaceImage.useConfig(backend=Backend.Torch)
        
        // Test with different data types
        let floatTensor = FurnaceImage.tensor([| 1.5f; 2.5f |], dtype=Dtype.Float32)
        let doubleTensor = FurnaceImage.tensor([| 1.5; 2.5 |], dtype=Dtype.Float64)
        let intTensor = FurnaceImage.tensor([| 1; 2 |], dtype=Dtype.Int32)
        
        // Convert to torch tensors
        let torchFloat = floatTensor.toTorch()
        let torchDouble = doubleTensor.toTorch()
        let torchInt = intTensor.toTorch()
        
        // Verify the torch tensors exist and have correct properties
        Assert.IsNotNull(torchFloat)
        Assert.IsNotNull(torchDouble)
        Assert.IsNotNull(torchInt)
        
        // Convert back and verify
        let reconstructedFloat = FurnaceImage.fromTorch(torchFloat)
        let reconstructedDouble = FurnaceImage.fromTorch(torchDouble)
        let reconstructedInt = FurnaceImage.fromTorch(torchInt)
        
        Assert.CheckEqual(Dtype.Float32, reconstructedFloat.dtype)
        Assert.CheckEqual(Dtype.Float64, reconstructedDouble.dtype)
        Assert.CheckEqual(Dtype.Int32, reconstructedInt.dtype)

    [<Test>]
    let TestTorchExtensionsToTorchWithNonTorchBackend() =
        use config = FurnaceImage.useConfig(backend=Backend.Reference, dtype=Dtype.Float32)
        
        // Create a tensor with Reference backend
        let referenceTensor = FurnaceImage.tensor([| 1.0f; 2.0f; 3.0f |])
        
        // Verify it's using Reference backend
        Assert.CheckEqual(Backend.Reference, referenceTensor.backend)
        
        // Attempting to convert to torch should throw an exception
        Assert.Throws<System.Exception>(fun () -> 
            let torchTensor = referenceTensor.toTorch()
            torchTensor |> ignore
        ) |> ignore

    [<Test>]
    let TestTorchExtensionsRoundTripLargerTensor() =
        use config = FurnaceImage.useConfig(backend=Backend.Torch, dtype=Dtype.Float32)
        
        // Create a larger 3D tensor
        let originalTensor = FurnaceImage.randn([2; 3; 4])
        
        // Convert to torch and back
        let torchTensor = originalTensor.toTorch()
        let reconstructedTensor = FurnaceImage.fromTorch(torchTensor)
        
        // Verify properties match
        Assert.CheckEqual(originalTensor.shape, reconstructedTensor.shape)
        Assert.CheckEqual(originalTensor.dtype, reconstructedTensor.dtype)
        Assert.CheckEqual(originalTensor.backend, reconstructedTensor.backend)
        Assert.CheckEqual(24, reconstructedTensor.nelement)  // 2 * 3 * 4 = 24
        
        // Verify values are preserved (3D tensor returns float32[,,])
        let originalArray = originalTensor.toArray() :?> float32[,,]
        let reconstructedArray = reconstructedTensor.toArray() :?> float32[,,]
        
        Assert.CheckEqual(originalArray.GetLength(0), reconstructedArray.GetLength(0))
        Assert.CheckEqual(originalArray.GetLength(1), reconstructedArray.GetLength(1))
        Assert.CheckEqual(originalArray.GetLength(2), reconstructedArray.GetLength(2))
        
        // Check values are preserved (allowing for floating point precision)
        for i in 0 .. originalArray.GetLength(0) - 1 do
            for j in 0 .. originalArray.GetLength(1) - 1 do
                for k in 0 .. originalArray.GetLength(2) - 1 do
                    Assert.AreEqual(float originalArray[i,j,k], float reconstructedArray[i,j,k], 1e-6)

    [<Test>]
    let TestTorchExtensionsBoolTensorRoundTrip() =
        use config = FurnaceImage.useConfig(backend=Backend.Torch, dtype=Dtype.Float32)
        
        // Create a boolean tensor (need to specify dtype explicitly)
        let originalTensor = FurnaceImage.tensor([| true; false; true; false |], dtype=Dtype.Bool)
        
        // Convert to torch and back
        let torchTensor = originalTensor.toTorch()
        let reconstructedTensor = FurnaceImage.fromTorch(torchTensor)
        
        // Verify properties
        Assert.CheckEqual(originalTensor.shape, reconstructedTensor.shape)
        Assert.CheckEqual(Dtype.Bool, reconstructedTensor.dtype)
        
        // Check boolean values are preserved
        let originalArray = originalTensor.toArray() :?> bool[]
        let reconstructedArray = reconstructedTensor.toArray() :?> bool[]
        
        Assert.CheckEqual(4, reconstructedArray.Length)
        Assert.CheckEqual(true, reconstructedArray[0])
        Assert.CheckEqual(false, reconstructedArray[1])
        Assert.CheckEqual(true, reconstructedArray[2])
        Assert.CheckEqual(false, reconstructedArray[3])