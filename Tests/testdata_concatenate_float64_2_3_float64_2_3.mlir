"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xf64>, tensor<2x3xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x3xf64>
    %5 = "stablehlo.concatenate"(%3#0, %3#1) <{dimension = 0 : i64}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<4x3xf64>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x3xf64>, tensor<4x3xf64>) -> ()
    "func.return"(%5) : (tensor<4x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xf64>, tensor<2x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1.3055351669265756, 5.359450440855829, -0.1245945850997244], [0.51025006358717651, -3.4455013123379579, 2.5991994918248387]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[-0.97707148751442662, 0.4321582786089882, -11.583640637044651], [-2.7751204424031188, -0.29065752456876909, -1.8843354920799062]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1, %2) : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.3055351669265756, 5.359450440855829, -0.1245945850997244], [0.51025006358717651, -3.4455013123379579, 2.5991994918248387], [-0.97707148751442662, 0.4321582786089882, -11.583640637044651], [-2.7751204424031188, -0.29065752456876909, -1.8843354920799062]]> : tensor<4x3xf64>}> : () -> tensor<4x3xf64>
    "func.return"(%0) : (tensor<4x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

