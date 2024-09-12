"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui16>, tensor<3x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui16>) -> tensor<4x3xf64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf64>) -> tensor<3x6xf64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    "func.return"(%7) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui16>, tensor<3x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4, 5, 1], [1, 1, 1], [1, 0, 1], [4, 3, 1]]> : tensor<4x3xui16>}> : () -> tensor<4x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[-7.3898443007284609, 0.19670499395453581, -1.7267214064489096, -5.5554539393575659, -4.3368963353758403, 1.4177000540488949], [-4.2871662468944489, 1.4031557079413877, -2.2210025607289889, 0.013673718816889249, 1.81969453715385, 1.804112442445156], [1.4987380528153229, 3.4588654567804267, 2.6412790852210235, -0.64853344011468306, 3.0950655806641274, -0.028863570797704914]]> : tensor<3x6xf64>}> : () -> tensor<3x6xf64>
    "func.return"(%1, %2) : (tensor<4x3xui16>, tensor<3x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-49.49647038457077, 11.261463972305508, -15.370619344219561, -22.8019806034605, -5.1540470750699834, 14.662498857623655], [-10.178272494807587, 5.0587261586763503, -1.306444881956875, -6.1903136606553595, 0.5778637824421371, 3.1929489256963461], [-5.8911062479131378, 3.6555704507349627, 0.91455767877211391, -6.2039873794722489, -1.2418307547117129, 1.3888364832511899], [-40.922137890781869, 8.4551525564227337, -10.928614222761581, -22.829328041094278, -8.7934361493776834, 11.054273972733343]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%0) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

