"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x4xf64>, tensor<3x4xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xcomplex<f64>>
    %5 = "stablehlo.complex"(%3#0, %3#1) : (tensor<3x4xf64>, tensor<3x4xf64>) -> tensor<3x4xcomplex<f64>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xcomplex<f64>>, tensor<3x4xcomplex<f64>>) -> ()
    "func.return"(%5) : (tensor<3x4xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x4xf64>, tensor<3x4xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2.2446081079638374, 2.9311191956533249, -0.80297253142501623, 1.3327935458596931], [-2.3572290064600101, -0.038978970864541988, 3.39249420391236, 1.7492081468625311], [7.1506041002014626, -4.5479984958432951, -5.7099287996096058, -2.1279896891312085]]> : tensor<3x4xf64>}> : () -> tensor<3x4xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[-4.7595820050969397, -1.6095319627071554, -0.39933304773392725, 5.1633630930350929], [0.95945138201202718, 0.6913424204045242, 0.11489614345782669, 1.0447167440530702], [1.3667622863058879, -0.55920604571652877, 3.4785332690751671, -2.2838064843987271]]> : tensor<3x4xf64>}> : () -> tensor<3x4xf64>
    "func.return"(%1, %2) : (tensor<3x4xf64>, tensor<3x4xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(2.2446081079638374,-4.7595820050969397), (2.9311191956533249,-1.6095319627071554), (-0.80297253142501623,-0.39933304773392725), (1.3327935458596931,5.1633630930350929)], [(-2.3572290064600101,0.95945138201202718), (-0.038978970864541988,0.6913424204045242), (3.39249420391236,0.11489614345782669), (1.7492081468625311,1.0447167440530702)], [(7.1506041002014626,1.3667622863058879), (-4.5479984958432951,-0.55920604571652877), (-5.7099287996096058,3.4785332690751671), (-2.1279896891312085,-2.2838064843987271)]]> : tensor<3x4xcomplex<f64>>}> : () -> tensor<3x4xcomplex<f64>>
    "func.return"(%0) : (tensor<3x4xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

