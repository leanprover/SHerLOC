"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1xbf16>, tensor<bf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      "stablehlo.return"(%arg1) : (tensor<bf16>) -> ()
    }) : (tensor<1xbf16>, tensor<1xi64>, tensor<bf16>) -> tensor<1xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1xbf16>, tensor<1xbf16>) -> ()
    "func.return"(%6) : (tensor<1xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1xbf16>, tensor<bf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<-1.359380e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %2 = "stablehlo.constant"() <{value = dense<-5.156250e-01> : tensor<bf16>}> : () -> tensor<bf16>
    "func.return"(%1, %2) : (tensor<1xbf16>, tensor<bf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<-5.156250e-01> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    "func.return"(%0) : (tensor<1xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

