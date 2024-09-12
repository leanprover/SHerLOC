"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1x3x1xi64>}> : () -> tensor<1x3x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xf64>, tensor<2x1x3xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<2x3xf64>, tensor<1x3x1xi64>, tensor<2x1x3xf64>) -> tensor<2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    "func.return"(%6) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xf64>, tensor<2x1x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-0.3356875496195324, -0.30514671473155519, 1.9629755390462527], [-1.9110390206632146, 3.0250865442748971, 0.047292113838280039]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[[0.2595826160226582, -4.2632347015298642, 3.2694717235449002]], [[2.6332022840408809, 0.46558743510622791, 1.0209954953443161]]]> : tensor<2x1x3xf64>}> : () -> tensor<2x1x3xf64>
    "func.return"(%1, %2) : (tensor<2x3xf64>, tensor<2x1x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-0.3356875496195324, -0.30514671473155519, -7.1024358631119373], [-1.9110390206632146, 3.0250865442748971, 0.059196772210423576]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%0) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

