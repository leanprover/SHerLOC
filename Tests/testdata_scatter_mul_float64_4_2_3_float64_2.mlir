"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf64>, tensor<2xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<4x2x3xf64>, tensor<2xi64>, tensor<2xf64>) -> tensor<4x2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf64>, tensor<4x2x3xf64>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf64>, tensor<2xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[4.4455846291335099, -4.3998732413367687, -0.15248406409536491], [8.4845700215915496, 0.23348036072342401, 0.28768682159009851]], [[0.93040811475067053, 0.82522037997738406, -4.7779635220780339], [-1.7991388580848229, 1.420174299847655, -2.7272671794621854]], [[-2.0668198282621644, -3.2068602469920697, -0.022292832446639686], [-4.8428876106641283, 4.1696060740179259, -2.4023229760249363]], [[-1.7745163750117949, -2.3888102691548441, 3.4787661413220592], [5.256327603370119, 3.4849275129792034, -1.3002171825159827]]]> : tensor<4x2x3xf64>}> : () -> tensor<4x2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[-0.31526834208474552, -3.4653195553336431]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%1, %2) : (tensor<4x2x3xf64>, tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[4.4455846291335099, -4.3998732413367687, -0.15248406409536491], [8.4845700215915496, 0.23348036072342401, 0.28768682159009851]], [[0.93040811475067053, 0.82522037997738406, -4.7779635220780339], [-1.7991388580848229, 1.420174299847655, -2.7272671794621854]], [[-2.0668198282621644, -3.2068602469920697, -0.022292832446639686], [-4.8428876106641283, 4.1696060740179259, -2.4023229760249363]], [[-1.7745163750117949, -2.3888102691548441, -1.0967448338751531], [5.256327603370119, 3.4849275129792034, 4.5056680287534476]]]> : tensor<4x2x3xf64>}> : () -> tensor<4x2x3xf64>
    "func.return"(%0) : (tensor<4x2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

