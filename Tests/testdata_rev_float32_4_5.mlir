"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x5xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x5xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<4x5xf32>
    %4 = "stablehlo.reverse"(%2) <{dimensions = array<i64: 0>}> : (tensor<4x5xf32>) -> tensor<4x5xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
    "func.return"(%4) : (tensor<4x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.953698933, 0.58084029, 2.857903, 3.98109102, 2.28901577], [-0.449118882, 0.0547672883, -2.05886245, 2.94395828, -3.02962065], [1.27594137, -2.0539515, -0.698254168, 2.72901058, -4.18458509], [-4.04998112, 2.12366033, -5.84264135, -1.21474063, 2.35050321]]> : tensor<4x5xf32>}> : () -> tensor<4x5xf32>
    "func.return"(%1) : (tensor<4x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.04998112, 2.12366033, -5.84264135, -1.21474063, 2.35050321], [1.27594137, -2.0539515, -0.698254168, 2.72901058, -4.18458509], [-0.449118882, 0.0547672883, -2.05886245, 2.94395828, -3.02962065], [0.953698933, 0.58084029, 2.857903, 3.98109102, 2.28901577]]> : tensor<4x5xf32>}> : () -> tensor<4x5xf32>
    "func.return"(%0) : (tensor<4x5xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

