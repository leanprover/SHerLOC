"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x2x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x3xf16>, tensor<2x3xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x2x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<1x2x3xf16>, tensor<1xi64>, tensor<2x3xf16>) -> tensor<1x2x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x2x3xf16>, tensor<1x2x3xf16>) -> ()
    "func.return"(%6) : (tensor<1x2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x3xf16>, tensor<2x3xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-1.794920e+00, -2.523440e+00, 3.210940e+00], [8.551030e-02, -2.320310e+00, 2.962890e+00]]]> : tensor<1x2x3xf16>}> : () -> tensor<1x2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[4.265630e+00, -1.109380e+00, 1.582030e+00], [-3.636720e+00, 1.430660e+00, 1.104490e+00]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    "func.return"(%1, %2) : (tensor<1x2x3xf16>, tensor<2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-7.656250e+00, 2.798830e+00, 5.078130e+00], [-3.110350e-01, -3.320310e+00, 3.273440e+00]]]> : tensor<1x2x3xf16>}> : () -> tensor<1x2x3xf16>
    "func.return"(%0) : (tensor<1x2x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

