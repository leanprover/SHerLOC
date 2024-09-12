"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf64>, tensor<2xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      "stablehlo.return"(%arg1) : (tensor<f64>) -> ()
    }) : (tensor<4x2x3xf64>, tensor<2xi64>, tensor<2xf64>) -> tensor<4x2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf64>, tensor<4x2x3xf64>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf64>, tensor<2xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-3.3343903567041435, 1.399443013955272, -0.88149822800170163], [3.7416562105176112, 1.4999008827278368, 7.1208132579317631]], [[-1.7940331284552999, -0.032175943684717788, 2.2352396108290491], [1.2186399212913772, 4.8434332112276506, 0.11716366357022928]], [[-5.3641866674376644, 0.40554360314447213, 4.9955057902746498], [-1.1143407702320949, -0.18456498185150411, 4.0132794740868114]], [[-0.25657870433660834, -0.072047263565402989, 5.3137062063393756], [3.0175286955646827, 0.70558779696416285, 2.7222660722662875]]]> : tensor<4x2x3xf64>}> : () -> tensor<4x2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[-4.4156708929267099, 2.3257735457850215]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%1, %2) : (tensor<4x2x3xf64>, tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-3.3343903567041435, 1.399443013955272, -0.88149822800170163], [3.7416562105176112, 1.4999008827278368, 7.1208132579317631]], [[-1.7940331284552999, -0.032175943684717788, 2.2352396108290491], [1.2186399212913772, 4.8434332112276506, 0.11716366357022928]], [[-5.3641866674376644, 0.40554360314447213, 4.9955057902746498], [-1.1143407702320949, -0.18456498185150411, 4.0132794740868114]], [[-0.25657870433660834, -0.072047263565402989, -4.4156708929267099], [3.0175286955646827, 0.70558779696416285, 2.3257735457850215]]]> : tensor<4x2x3xf64>}> : () -> tensor<4x2x3xf64>
    "func.return"(%0) : (tensor<4x2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

