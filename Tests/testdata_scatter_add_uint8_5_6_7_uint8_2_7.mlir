"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui8>, tensor<2x7xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<5x6x7xui8>, tensor<2x2xi64>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui8>, tensor<2x7xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x02030105020102030100000105030001000001030205000302010004020401020100000201010105020302040602000202000103000300020302060204020100020207020200020602000102060301010409000200010300050101050000000101010101030004010104000304020100080001000302000507000100020502050400020402030107000003000004030B010104000600000001080100050102010101000200000002010004000001000004030200060102000502040301010002000001020101000804090002000000000602"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 5, 5, 0, 0, 6, 3], [3, 0, 1, 1, 2, 1, 1]]> : tensor<2x7xui8>}> : () -> tensor<2x7xui8>
    "func.return"(%1, %2) : (tensor<5x6x7xui8>, tensor<2x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0203010502010204060500010B060001000001030205000302010004020401020100000201010105020302040602000202000103000300020302060204020100020207020200020602000102060301010409000200010300050101050000000101010101030004010107000405040201080001000302000507000100020502050400020402030107000003000004030B010104000600000001080100050102010101000200000002010004000001000004030200060102000502040301010002000001020101000804090002000000000602"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    "func.return"(%0) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

