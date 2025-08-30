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
type TestPyplot() =

    [<Test>]
    member _.TestPyplotConstructorDefaults() =
        // Test Pyplot constructor with default parameters
        let plt = Pyplot()
        Assert.IsNotNull(plt)

    [<Test>]
    member _.TestPyplotConstructorCustomParameters() =
        // Test Pyplot constructor with custom parameters
        let plt = Pyplot(pythonExecutable="python3", timeoutMilliseconds=5000)
        Assert.IsNotNull(plt)

    [<Test>]
    member _.TestPyplotFigure() =
        // Test figure method
        let plt = Pyplot()
        Assert.DoesNotThrow(fun () -> plt.figure())

    [<Test>]
    member _.TestPyplotFigureWithSize() =
        // Test figure method with size
        let plt = Pyplot()
        Assert.DoesNotThrow(fun () -> plt.figure((10.0, 6.0)))

    [<Test>]
    member _.TestPyplotPlotTensor() =
        // Test plot method with tensor
        let plt = Pyplot()
        let data = FurnaceImage.tensor([1.0f; 2.0f; 3.0f; 2.0f; 1.0f])
        Assert.DoesNotThrow(fun () -> plt.plot(data))

    [<Test>]
    member _.TestPyplotPlotTensorWithLabel() =
        // Test plot method with tensor and label
        let plt = Pyplot()
        let data = FurnaceImage.tensor([1.0f; 2.0f; 3.0f])
        Assert.DoesNotThrow(fun () -> plt.plot(data, label="test data"))

    [<Test>]
    member _.TestPyplotPlotTensorWithAlpha() =
        // Test plot method with tensor and alpha
        let plt = Pyplot()
        let data = FurnaceImage.tensor([1.0f; 2.0f; 3.0f])
        Assert.DoesNotThrow(fun () -> plt.plot(data, alpha=0.7))

    [<Test>]
    member _.TestPyplotPlotTensorWithLabelAndAlpha() =
        // Test plot method with tensor, label and alpha
        let plt = Pyplot()
        let data = FurnaceImage.tensor([1.0f; 2.0f; 3.0f])
        Assert.DoesNotThrow(fun () -> plt.plot(data, label="test", alpha=0.5))

    [<Test>]
    member _.TestPyplotPlotXYTensors() =
        // Test plot method with x and y tensors
        let plt = Pyplot()
        let x = FurnaceImage.tensor([1.0f; 2.0f; 3.0f])
        let y = FurnaceImage.tensor([2.0f; 4.0f; 6.0f])
        Assert.DoesNotThrow(fun () -> plt.plot(x, y))

    [<Test>]
    member _.TestPyplotPlotXYTensorsWithLabel() =
        // Test plot method with x, y tensors and label
        let plt = Pyplot()
        let x = FurnaceImage.tensor([1.0f; 2.0f; 3.0f])
        let y = FurnaceImage.tensor([2.0f; 4.0f; 6.0f])
        Assert.DoesNotThrow(fun () -> plt.plot(x, y, label="linear"))

    [<Test>]
    member _.TestPyplotPlotXYTensorsWithAlpha() =
        // Test plot method with x, y tensors and alpha
        let plt = Pyplot()
        let x = FurnaceImage.tensor([1.0f; 2.0f; 3.0f])
        let y = FurnaceImage.tensor([2.0f; 4.0f; 6.0f])
        Assert.DoesNotThrow(fun () -> plt.plot(x, y, alpha=0.8))

    [<Test>]
    member _.TestPyplotPlotXYTensorsWithLabelAndAlpha() =
        // Test plot method with x, y tensors, label and alpha
        let plt = Pyplot()
        let x = FurnaceImage.tensor([1.0f; 2.0f; 3.0f])
        let y = FurnaceImage.tensor([2.0f; 4.0f; 6.0f])
        Assert.DoesNotThrow(fun () -> plt.plot(x, y, label="data", alpha=0.6))

    [<Test>]
    member _.TestPyplotHistTensor() =
        // Test hist method with tensor
        let plt = Pyplot()
        let data = FurnaceImage.randn([100])
        Assert.DoesNotThrow(fun () -> plt.hist(data))

    [<Test>]
    member _.TestPyplotHistTensorWithBins() =
        // Test hist method with tensor and bins
        let plt = Pyplot()
        let data = FurnaceImage.randn([50])
        Assert.DoesNotThrow(fun () -> plt.hist(data, bins=20))

    [<Test>]
    member _.TestPyplotHistTensorWithLabel() =
        // Test hist method with tensor and label
        let plt = Pyplot()
        let data = FurnaceImage.randn([30])
        Assert.DoesNotThrow(fun () -> plt.hist(data, label="test data"))

    [<Test>]
    member _.TestPyplotHistTensorWithBinsAndLabel() =
        // Test hist method with tensor, bins and label
        let plt = Pyplot()
        let data = FurnaceImage.randn([40])
        Assert.DoesNotThrow(fun () -> plt.hist(data, bins=15, label="histogram"))

    [<Test>]
    member _.TestPyplotXlabelYlabel() =
        // Test xlabel and ylabel methods
        let plt = Pyplot()
        Assert.DoesNotThrow(fun () -> plt.xlabel("X Axis"))
        Assert.DoesNotThrow(fun () -> plt.ylabel("Y Axis"))

    [<Test>]
    member _.TestPyplotLegend() =
        // Test legend method
        let plt = Pyplot()
        Assert.DoesNotThrow(fun () -> plt.legend())

    [<Test>]
    member _.TestPyplotTightLayout() =
        // Test tightLayout method
        let plt = Pyplot()
        Assert.DoesNotThrow(fun () -> plt.tightLayout())

    [<Test>]
    member _.TestPyplotSavefig() =
        // Test savefig method - should not throw but may print warning
        let plt = Pyplot()
        let tempFile = Path.GetTempFileName() + ".png"
        Assert.DoesNotThrow(fun () -> plt.savefig(tempFile))

    [<Test>]
    member _.TestPyplotCompleteWorkflow() =
        // Test complete plotting workflow
        let plt = Pyplot()
        let x = FurnaceImage.tensor([1.0f; 2.0f; 3.0f; 4.0f; 5.0f])
        let y = FurnaceImage.tensor([1.0f; 4.0f; 9.0f; 16.0f; 25.0f])
        
        Assert.DoesNotThrow(fun () ->
            plt.figure((8.0, 6.0))
            plt.plot(x, y, label="quadratic", alpha=0.8)
            plt.xlabel("X values")
            plt.ylabel("Y values") 
            plt.legend()
            plt.tightLayout()
            let tempFile = Path.GetTempFileName() + ".png"
            plt.savefig(tempFile)
        )