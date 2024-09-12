"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x2x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[1, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x3xbf16>, tensor<1xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x2x3xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<1x2x3xbf16>, tensor<2xi64>, tensor<1xbf16>) -> tensor<1x2x3xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x2x3xbf16>, tensor<1x2x3xbf16>) -> ()
    "func.return"(%6) : (tensor<1x2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x3xbf16>, tensor<1xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[2.859380e+00, -6.000000e+00, 2.484380e+00], [-2.421880e+00, 1.210940e+00, 1.542970e-01]]]> : tensor<1x2x3xbf16>}> : () -> tensor<1x2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<-1.687500e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    "func.return"(%1, %2) : (tensor<1x2x3xbf16>, tensor<1xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[2.859380e+00, -6.000000e+00, 2.484380e+00], [-2.421880e+00, 1.210940e+00, 1.542970e-01]]]> : tensor<1x2x3xbf16>}> : () -> tensor<1x2x3xbf16>
    "func.return"(%0) : (tensor<1x2x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

