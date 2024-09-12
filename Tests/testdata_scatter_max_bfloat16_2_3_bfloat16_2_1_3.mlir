"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1x3x1xi64>}> : () -> tensor<1x3x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xbf16>, tensor<2x1x3xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<2x3xbf16>, tensor<1x3x1xi64>, tensor<2x1x3xbf16>) -> tensor<2x3xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
    "func.return"(%6) : (tensor<2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xbf16>, tensor<2x1x3xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.990720e-02, 2.171880e+00, -1.523440e-01], [-1.078130e+00, 1.281250e+00, -7.890630e-01]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[2.437500e+00, -1.234380e+00, -3.406250e+00]], [[3.421880e+00, 3.265630e+00, -2.781250e+00]]]> : tensor<2x1x3xbf16>}> : () -> tensor<2x1x3xbf16>
    "func.return"(%1, %2) : (tensor<2x3xbf16>, tensor<2x1x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.990720e-02, 2.171880e+00, 2.437500e+00], [-1.078130e+00, 1.281250e+00, 3.421880e+00]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    "func.return"(%0) : (tensor<2x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

