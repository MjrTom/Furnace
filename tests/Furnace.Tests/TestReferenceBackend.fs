// Copyright (c) 2016-     University of Oxford (Atılım Güneş Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System
open NUnit.Framework
open Furnace

[<TestFixture>]
type TestReferenceBackend () =

    [<Test>]
    member _.TestReferenceBackendFloat16Operations () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float16)
        
        // Basic tensor operations for Float16
        let t1 = combo.tensor([1.0f; 2.0f; 3.0f])
        let t2 = combo.tensor([2.0f; 4.0f; 6.0f])
        
        // Test arithmetic operations
        let addResult = t1 + t2
        let expectedAdd = combo.tensor([3.0f; 6.0f; 9.0f])
        Assert.True(expectedAdd.allclose(addResult, 0.001))
        
        let subResult = t2 - t1
        let expectedSub = combo.tensor([1.0f; 2.0f; 3.0f])
        Assert.True(expectedSub.allclose(subResult, 0.001))
        
        let mulResult = t1 * t2
        let expectedMul = combo.tensor([2.0f; 8.0f; 18.0f])
        Assert.True(expectedMul.allclose(mulResult, 0.001))
        
        let divResult = t2 / t1
        let expectedDiv = combo.tensor([2.0f; 2.0f; 2.0f])
        Assert.True(expectedDiv.allclose(divResult, 0.001))

    [<Test>]
    member _.TestReferenceBackendBFloat16Operations () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.BFloat16)
        
        // Basic tensor operations for BFloat16
        let t1 = combo.tensor([1.0f; 2.0f; 3.0f])
        let t2 = combo.tensor([4.0f; 5.0f; 6.0f])
        
        // Test comparison operations  
        let ltResult = t1.lt(t2)
        let expectedLt = FurnaceImage.tensor([true; true; true], dtype=Dtype.Bool, device=combo.device, backend=combo.backend)
        Assert.CheckEqual(expectedLt, ltResult)
        
        let gtResult = t1.gt(t2)
        let expectedGt = FurnaceImage.tensor([false; false; false], dtype=Dtype.Bool, device=combo.device, backend=combo.backend)
        Assert.CheckEqual(expectedGt, gtResult)

    [<Test>]
    member _.TestReferenceBackendBoolOperations () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Bool)
        
        // Test boolean tensor operations
        let t1 = combo.tensor([true; false; true])
        let t2 = combo.tensor([false; true; true])
        
        // Test basic properties
        Assert.AreEqual(Dtype.Bool, t1.dtype)
        Assert.AreEqual(Backend.Reference, t1.backend)
        
        // Test element access
        let item0 = t1[0].toScalar()
        let item1 = t1[1].toScalar()
        let item2 = t1[2].toScalar()
        Assert.AreEqual(1.0, item0.toDouble())  // true converted to 1.0
        Assert.AreEqual(0.0, item1.toDouble())  // false converted to 0.0
        Assert.AreEqual(1.0, item2.toDouble())  // true converted to 1.0

    [<Test>]
    member _.TestReferenceBackendFloat16MatrixOperations () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float16)
        
        // Test 2D operations for Float16
        let m1 = combo.tensor([[1.0f; 2.0f]; [3.0f; 4.0f]])
        let m2 = combo.tensor([[2.0f; 0.0f]; [1.0f; 2.0f]])
        
        // Matrix multiplication
        let matmulResult = m1.matmul(m2)
        let expectedMatmul = combo.tensor([[4.0f; 4.0f]; [10.0f; 8.0f]])
        Assert.True(expectedMatmul.allclose(matmulResult, 0.001))
        
        // Transpose
        let transposeResult = m1.transpose()
        let expectedTranspose = combo.tensor([[1.0f; 3.0f]; [2.0f; 4.0f]])
        Assert.True(expectedTranspose.allclose(transposeResult, 0.001))

    [<Test>]
    member _.TestReferenceBackendBFloat16ReductionOperations () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.BFloat16)
        
        // Test reduction operations for BFloat16
        let t = combo.tensor([[1.0f; 2.0f; 3.0f]; [4.0f; 5.0f; 6.0f]])
        
        // Sum operations
        let sumResult = t.sum()
        Assert.True(abs(sumResult.toScalar().toSingle() - 21.0f) < 0.001f)
        
        let sumDim0 = t.sum(0)
        let expectedSumDim0 = combo.tensor([5.0f; 7.0f; 9.0f])
        Assert.True(expectedSumDim0.allclose(sumDim0, 0.001))
        
        let sumDim1 = t.sum(1)
        let expectedSumDim1 = combo.tensor([6.0f; 15.0f])
        Assert.True(expectedSumDim1.allclose(sumDim1, 0.001))

    [<Test>]
    member _.TestReferenceBackendFloat16IndexingOperations () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float16)
        
        // Test indexing and slicing operations
        let t = combo.tensor([1.0f; 2.0f; 3.0f; 4.0f; 5.0f])
        
        // Single element indexing
        let item = t[2]
        Assert.True(abs(item.toScalar().toSingle() - 3.0f) < 0.001f)
        
        // Slice operations
        let slice = t[1..3]
        let expectedSlice = combo.tensor([2.0f; 3.0f; 4.0f])
        Assert.True(expectedSlice.allclose(slice, 0.001))

    [<Test>]
    member _.TestReferenceBackendMixedTypeOperations () =
        let comboFloat16 = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float16)
        let comboBFloat16 = ComboInfo(Backend.Reference, Device.CPU, Dtype.BFloat16)
        
        // Test operations between different precisions
        let t1 = comboFloat16.tensor([1.0f; 2.0f])
        let t2 = comboBFloat16.tensor([3.0f; 4.0f])
        
        // These should work through type coercion
        let result = t1.cast(Dtype.Float32) + t2.cast(Dtype.Float32)
        let expected = FurnaceImage.tensor([4.0f; 6.0f], dtype=Dtype.Float32, backend=Backend.Reference)
        Assert.True(expected.allclose(result, 0.001))

    [<Test>]
    member _.TestReferenceBackendActivationFunctions () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float16)
        
        // Test activation functions on Float16
        let t = combo.tensor([-2.0f; -1.0f; 0.0f; 1.0f; 2.0f])
        
        // Sigmoid
        let sigmoidResult = t.sigmoid()
        Assert.AreEqual(5, sigmoidResult.nelement)
        
        // Tanh
        let tanhResult = t.tanh()
        Assert.AreEqual(5, tanhResult.nelement)
        
        // ReLU
        let reluResult = t.relu()
        let expectedRelu = combo.tensor([0.0f; 0.0f; 0.0f; 1.0f; 2.0f])
        Assert.True(expectedRelu.allclose(reluResult, 0.001))

    [<Test>]
    member _.TestReferenceBackendShapeOperations () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.BFloat16)
        
        // Test shape manipulation operations
        let t = combo.tensor([1.0f; 2.0f; 3.0f; 4.0f; 5.0f; 6.0f])
        
        // Reshape
        let reshaped = t.view([2; 3])
        Assert.AreEqual([|2; 3|], reshaped.shape)
        
        // Flatten
        let flattened = reshaped.view([-1])
        Assert.AreEqual([|6|], flattened.shape)
        
        // Squeeze/unsqueeze
        let unsqueezed = t.unsqueeze(0)
        Assert.AreEqual([|1; 6|], unsqueezed.shape)
        
        let squeezed = unsqueezed.squeeze(0)
        Assert.AreEqual([|6|], squeezed.shape)

    [<Test>]
    member _.TestReferenceBackendEdgeCases () =
        // Test edge cases for lower precision types
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float16)
        
        // Empty tensors (create with shape)
        let empty = FurnaceImage.zeros([0], dtype=combo.dtype, backend=combo.backend)
        Assert.AreEqual(0, empty.nelement)
        Assert.AreEqual([|0|], empty.shape)
        
        // Single element tensors
        let single = combo.tensor([42.0f])
        Assert.True(abs(single.toScalar().toSingle() - 42.0f) < 0.001f)
        
        // Large tensors for stress testing
        let large = combo.randn([100; 50])
        Assert.AreEqual([|100; 50|], large.shape)
        Assert.AreEqual(5000, large.nelement)

    [<Test>]
    member _.TestReferenceBackendFloat16ActivationDerivatives () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.Float16)
        
        // Test activation function operations that might not be well covered
        let t = combo.tensor([0.5f; 1.0f; 1.5f])
        
        // Test softplus (might be less covered)
        let softplusResult = t.softplus()
        Assert.AreEqual(3, softplusResult.nelement)
        Assert.AreEqual(Dtype.Float16, softplusResult.dtype)
        
        // Test exp and log operations
        let expResult = t.exp()
        Assert.AreEqual(3, expResult.nelement)
        Assert.AreEqual(Dtype.Float16, expResult.dtype)
        
        let logResult = t.log()
        Assert.AreEqual(3, logResult.nelement)
        Assert.AreEqual(Dtype.Float16, logResult.dtype)

    [<Test>]
    member _.TestReferenceBackendBFloat16ComparisonEdgeCases () =
        let combo = ComboInfo(Backend.Reference, Device.CPU, Dtype.BFloat16)
        
        // Test edge cases in comparison operations  
        let t1 = combo.tensor([1.0f; 2.0f; 3.0f])
        let t2 = combo.tensor([1.0f; 1.5f; 4.0f])
        
        // Test less than or equal
        let leResult = t1.le(t2)
        let expectedLe = FurnaceImage.tensor([true; false; true], dtype=Dtype.Bool, device=combo.device, backend=combo.backend)
        Assert.CheckEqual(expectedLe, leResult)
        
        // Test greater than or equal  
        let geResult = t1.ge(t2)
        let expectedGe = FurnaceImage.tensor([true; true; false], dtype=Dtype.Bool, device=combo.device, backend=combo.backend)
        Assert.CheckEqual(expectedGe, geResult)
        
        // Test not equal (using .ne instead of .neq)
        let neResult = t1.ne(t2)
        let expectedNe = FurnaceImage.tensor([false; true; true], dtype=Dtype.Bool, device=combo.device, backend=combo.backend)
        Assert.CheckEqual(expectedNe, neResult)