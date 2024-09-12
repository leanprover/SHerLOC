"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xf64>
    %4 = "stablehlo.transpose"(%2) <{permutation = array<i64: 1, 0>}> : (tensor<2x3xf64>) -> tensor<3x2xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xf64>, tensor<3x2xf64>) -> ()
    "func.return"(%4) : (tensor<3x2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.3456966792229577, 5.3498934080825755, 1.5772425443498772], [0.34120593540777594, -1.8904489264669952, 2.6353488633822506]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.3456966792229577, 0.34120593540777594], [5.3498934080825755, -1.8904489264669952], [1.5772425443498772, 2.6353488633822506]]> : tensor<3x2xf64>}> : () -> tensor<3x2xf64>
    "func.return"(%0) : (tensor<3x2xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

