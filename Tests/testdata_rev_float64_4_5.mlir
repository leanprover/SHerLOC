"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x5xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x5xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<4x5xf64>
    %4 = "stablehlo.reverse"(%2) <{dimensions = array<i64: 0>}> : (tensor<4x5xf64>) -> tensor<4x5xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x5xf64>, tensor<4x5xf64>) -> ()
    "func.return"(%4) : (tensor<4x5xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-0.068293202679957851, -2.6867475929167197, -2.2749228292440544, 3.5248234665276428, -0.11431189492327372], [1.6962983012984216, -7.0719194750213958, -0.32469278565706106, -3.2470842711756642, -10.343215461700757], [0.8597751311741596, -1.4606133085628474, 0.95919580967951789, 1.4313565361388514, 2.9807530678976066], [2.9634058562791141, 3.6780890335969096, -4.0214880095131598, 1.8323414945632726, 4.4693645223023495]]> : tensor<4x5xf64>}> : () -> tensor<4x5xf64>
    "func.return"(%1) : (tensor<4x5xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[2.9634058562791141, 3.6780890335969096, -4.0214880095131598, 1.8323414945632726, 4.4693645223023495], [0.8597751311741596, -1.4606133085628474, 0.95919580967951789, 1.4313565361388514, 2.9807530678976066], [1.6962983012984216, -7.0719194750213958, -0.32469278565706106, -3.2470842711756642, -10.343215461700757], [-0.068293202679957851, -2.6867475929167197, -2.2749228292440544, 3.5248234665276428, -0.11431189492327372]]> : tensor<4x5xf64>}> : () -> tensor<4x5xf64>
    "func.return"(%0) : (tensor<4x5xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

