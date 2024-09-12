"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xf32>, tensor<2x3xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x6xf32>
    %5 = "stablehlo.concatenate"(%3#0, %3#1) <{dimension = 1 : i64}> : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x6xf32>, tensor<2x6xf32>) -> ()
    "func.return"(%5) : (tensor<2x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xf32>, tensor<2x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.29547954, 2.90465164, -6.267250e+00], [1.80744612, -0.891328752, -5.21437025]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[2.62669659, 0.361944199, -6.23034668], [5.21641445, 0.966221809, 3.38860106]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    "func.return"(%1, %2) : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.29547954, 2.90465164, -6.267250e+00, 2.62669659, 0.361944199, -6.23034668], [1.80744612, -0.891328752, -5.21437025, 5.21641445, 0.966221809, 3.38860106]]> : tensor<2x6xf32>}> : () -> tensor<2x6xf32>
    "func.return"(%0) : (tensor<2x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

