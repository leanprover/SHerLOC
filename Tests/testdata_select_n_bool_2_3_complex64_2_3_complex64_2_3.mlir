"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xcomplex<f32>>
    %6 = "stablehlo.select"(%4#0, %4#2, %4#1) : (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
    "func.return"(%6) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    %2 = "stablehlo.constant"() <{value = dense<[[(2.43387675,4.23526859), (1.10274744,2.2652235), (-4.38410425,0.678857446)], [(0.126821086,-3.64852381), (-1.20231128,3.67738676), (3.78677344,4.46107054)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    %3 = "stablehlo.constant"() <{value = dense<[[(3.70096087,0.259771079), (-2.55815077,0.15060997), (1.94375587,1.97517061)], [(-6.01416063,-2.1360147), (2.66556025,3.83397913), (-2.5734024,-1.7117784)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%1, %2, %3) : (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(3.70096087,0.259771079), (-2.55815077,0.15060997), (1.94375587,1.97517061)], [(-6.01416063,-2.1360147), (2.66556025,3.83397913), (-2.5734024,-1.7117784)]]> : tensor<2x3xcomplex<f32>>}> : () -> tensor<2x3xcomplex<f32>>
    "func.return"(%0) : (tensor<2x3xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

