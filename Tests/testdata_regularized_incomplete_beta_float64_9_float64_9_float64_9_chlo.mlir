"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<9xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %50:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<9xf64>, tensor<9xf64>, tensor<9xf64>)
    %51 = "func.call"() <{callee = @expected}> : () -> tensor<9xf64>
    %52 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %53 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %54 = "stablehlo.compare"(%50#0, %53) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
    %55 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %56 = "stablehlo.broadcast_in_dim"(%55) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %57 = "stablehlo.compare"(%50#1, %56) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
    %58 = "stablehlo.or"(%54, %57) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %59 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %60 = "stablehlo.broadcast_in_dim"(%59) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %61 = "stablehlo.compare"(%50#2, %60) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
    %62 = "stablehlo.or"(%58, %61) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %63 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %64 = "stablehlo.broadcast_in_dim"(%63) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %65 = "stablehlo.compare"(%50#2, %64) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
    %66 = "stablehlo.or"(%62, %65) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %67 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %68 = "stablehlo.broadcast_in_dim"(%67) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %69 = "stablehlo.add"(%50#0, %68) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %70 = "stablehlo.add"(%50#0, %50#1) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %71 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %72 = "stablehlo.broadcast_in_dim"(%71) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %73 = "stablehlo.add"(%70, %72) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %74 = "stablehlo.divide"(%69, %73) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %75 = "stablehlo.compare"(%50#2, %74) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
    %76 = "stablehlo.select"(%75, %50#0, %50#1) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %77 = "stablehlo.select"(%75, %50#1, %50#0) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %78 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %79 = "stablehlo.broadcast_in_dim"(%78) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %80 = "stablehlo.subtract"(%79, %50#2) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %81 = "stablehlo.select"(%75, %50#2, %80) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %82 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %83 = "stablehlo.broadcast_in_dim"(%82) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %84 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %85 = "stablehlo.broadcast_in_dim"(%84) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %86 = "stablehlo.compare"(%83, %85) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %87 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %88 = "stablehlo.broadcast_in_dim"(%87) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %89 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %90 = "stablehlo.broadcast_in_dim"(%89) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %91 = "stablehlo.select"(%86, %88, %90) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %92 = "stablehlo.abs"(%91) : (tensor<9xf64>) -> tensor<9xf64>
    %93 = "stablehlo.constant"() <{value = dense<1.1102230246251565E-16> : tensor<f64>}> : () -> tensor<f64>
    %94 = "stablehlo.broadcast_in_dim"(%93) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %95 = "stablehlo.compare"(%92, %94) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
    %96 = "stablehlo.constant"() <{value = dense<1.1102230246251565E-16> : tensor<f64>}> : () -> tensor<f64>
    %97 = "stablehlo.broadcast_in_dim"(%96) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %98 = "stablehlo.select"(%95, %97, %91) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %99 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %100 = "stablehlo.broadcast_in_dim"(%99) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %101 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %102 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %103:8 = "stablehlo.while"(%76, %77, %81, %101, %102, %98, %100, %98) ({
    ^bb0(%arg21: tensor<9xf64>, %arg22: tensor<9xf64>, %arg23: tensor<9xf64>, %arg24: tensor<i64>, %arg25: tensor<i1>, %arg26: tensor<9xf64>, %arg27: tensor<9xf64>, %arg28: tensor<9xf64>):
      %211 = "stablehlo.constant"() <{value = dense<600> : tensor<i64>}> : () -> tensor<i64>
      %212 = "stablehlo.compare"(%arg24, %211) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %213 = "stablehlo.and"(%212, %arg25) : (tensor<i1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%213) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg11: tensor<9xf64>, %arg12: tensor<9xf64>, %arg13: tensor<9xf64>, %arg14: tensor<i64>, %arg15: tensor<i1>, %arg16: tensor<9xf64>, %arg17: tensor<9xf64>, %arg18: tensor<9xf64>):
      %127 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %128 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
      %129 = "stablehlo.broadcast_in_dim"(%128) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %130 = "func.call"(%127, %129) <{callee = @remainder}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %131 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %132 = "stablehlo.broadcast_in_dim"(%131) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %133 = "stablehlo.compare"(%130, %132) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %134 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %135 = "stablehlo.broadcast_in_dim"(%134) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %136 = "stablehlo.compare"(%127, %135) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %137 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %138 = "stablehlo.broadcast_in_dim"(%137) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %139 = "stablehlo.subtract"(%127, %138) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %140 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
      %141 = "stablehlo.broadcast_in_dim"(%140) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %142 = "func.call"(%139, %141) <{callee = @floor_divide}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
      %143 = "stablehlo.convert"(%142) : (tensor<9xi64>) -> tensor<9xf64>
      %144 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %145 = "stablehlo.broadcast_in_dim"(%144) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
      %146 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %147 = "stablehlo.broadcast_in_dim"(%146) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
      %148 = "stablehlo.add"(%arg11, %143) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %149 = "stablehlo.negate"(%148) : (tensor<9xf64>) -> tensor<9xf64>
      %150 = "stablehlo.add"(%arg11, %arg12) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %151 = "stablehlo.add"(%150, %143) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %152 = "stablehlo.multiply"(%149, %151) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %153 = "stablehlo.multiply"(%152, %arg13) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %154 = "stablehlo.multiply"(%147, %143) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %155 = "stablehlo.add"(%arg11, %154) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %156 = "stablehlo.multiply"(%147, %143) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %157 = "stablehlo.add"(%arg11, %156) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %158 = "stablehlo.add"(%157, %145) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %159 = "stablehlo.multiply"(%155, %158) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %160 = "stablehlo.divide"(%153, %159) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %161 = "stablehlo.subtract"(%arg12, %143) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %162 = "stablehlo.multiply"(%143, %161) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %163 = "stablehlo.multiply"(%162, %arg13) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %164 = "stablehlo.multiply"(%147, %143) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %165 = "stablehlo.add"(%arg11, %164) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %166 = "stablehlo.subtract"(%165, %145) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %167 = "stablehlo.multiply"(%147, %143) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %168 = "stablehlo.add"(%arg11, %167) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %169 = "stablehlo.multiply"(%166, %168) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %170 = "stablehlo.divide"(%163, %169) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %171 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %172 = "stablehlo.broadcast_in_dim"(%171) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
      %173 = "stablehlo.select"(%133, %160, %170) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %174 = "stablehlo.select"(%136, %172, %173) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %175 = "stablehlo.broadcast_in_dim"(%arg14) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %176 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
      %177 = "stablehlo.broadcast_in_dim"(%176) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
      %178 = "stablehlo.compare"(%175, %177) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
      %179 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %180 = "stablehlo.broadcast_in_dim"(%179) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
      %181 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %182 = "stablehlo.broadcast_in_dim"(%181) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
      %183 = "stablehlo.select"(%178, %180, %182) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %184 = "stablehlo.divide"(%174, %arg16) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %185 = "stablehlo.add"(%183, %184) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %186 = "stablehlo.constant"() <{value = dense<1.1102230246251565E-16> : tensor<f64>}> : () -> tensor<f64>
      %187 = "stablehlo.broadcast_in_dim"(%186) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
      %188 = "stablehlo.abs"(%185) : (tensor<9xf64>) -> tensor<9xf64>
      %189 = "stablehlo.compare"(%188, %187) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
      %190 = "stablehlo.select"(%189, %187, %185) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %191 = "stablehlo.multiply"(%174, %arg17) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %192 = "stablehlo.add"(%183, %191) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %193 = "stablehlo.abs"(%192) : (tensor<9xf64>) -> tensor<9xf64>
      %194 = "stablehlo.compare"(%193, %187) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
      %195 = "stablehlo.select"(%194, %187, %192) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %196 = "func.call"(%195) <{callee = @integer_pow}> : (tensor<9xf64>) -> tensor<9xf64>
      %197 = "stablehlo.multiply"(%190, %196) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %198 = "stablehlo.multiply"(%arg18, %197) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %199 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %200 = "stablehlo.add"(%arg14, %199) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %201 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
      %202 = "stablehlo.broadcast_in_dim"(%201) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
      %203 = "stablehlo.subtract"(%197, %202) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
      %204 = "stablehlo.abs"(%203) : (tensor<9xf64>) -> tensor<9xf64>
      %205 = "stablehlo.constant"() <{value = dense<1.1102230246251565E-16> : tensor<f64>}> : () -> tensor<f64>
      %206 = "stablehlo.broadcast_in_dim"(%205) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
      %207 = "stablehlo.compare"(%204, %206) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xi1>
      %208 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
      %209 = "stablehlo.reduce"(%207, %208) <{dimensions = array<i64: 0>}> ({
      ^bb0(%arg19: tensor<i1>, %arg20: tensor<i1>):
        %210 = "stablehlo.or"(%arg19, %arg20) : (tensor<i1>, tensor<i1>) -> tensor<i1>
        "stablehlo.return"(%210) : (tensor<i1>) -> ()
      }) : (tensor<9xi1>, tensor<i1>) -> tensor<i1>
      "stablehlo.return"(%arg11, %arg12, %arg13, %200, %209, %190, %196, %198) : (tensor<9xf64>, tensor<9xf64>, tensor<9xf64>, tensor<i64>, tensor<i1>, tensor<9xf64>, tensor<9xf64>, tensor<9xf64>) -> ()
    }) : (tensor<9xf64>, tensor<9xf64>, tensor<9xf64>, tensor<i64>, tensor<i1>, tensor<9xf64>, tensor<9xf64>, tensor<9xf64>) -> (tensor<9xf64>, tensor<9xf64>, tensor<9xf64>, tensor<i64>, tensor<i1>, tensor<9xf64>, tensor<9xf64>, tensor<9xf64>)
    %104 = "chlo.lgamma"(%76) : (tensor<9xf64>) -> tensor<9xf64>
    %105 = "chlo.lgamma"(%77) : (tensor<9xf64>) -> tensor<9xf64>
    %106 = "stablehlo.add"(%104, %105) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %107 = "stablehlo.add"(%76, %77) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %108 = "chlo.lgamma"(%107) : (tensor<9xf64>) -> tensor<9xf64>
    %109 = "stablehlo.subtract"(%106, %108) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %110 = "stablehlo.log"(%81) : (tensor<9xf64>) -> tensor<9xf64>
    %111 = "stablehlo.multiply"(%110, %76) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %112 = "stablehlo.negate"(%81) : (tensor<9xf64>) -> tensor<9xf64>
    %113 = "stablehlo.log_plus_one"(%112) : (tensor<9xf64>) -> tensor<9xf64>
    %114 = "stablehlo.multiply"(%113, %77) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %115 = "stablehlo.add"(%111, %114) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %116 = "stablehlo.subtract"(%115, %109) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %117 = "stablehlo.exponential"(%116) : (tensor<9xf64>) -> tensor<9xf64>
    %118 = "stablehlo.multiply"(%103#7, %117) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %119 = "stablehlo.divide"(%118, %76) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %120 = "stablehlo.constant"() <{value = dense<0x7FF8000000000000> : tensor<f64>}> : () -> tensor<f64>
    %121 = "stablehlo.broadcast_in_dim"(%120) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %122 = "stablehlo.select"(%66, %121, %119) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %123 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %124 = "stablehlo.broadcast_in_dim"(%123) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %125 = "stablehlo.subtract"(%124, %122) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    %126 = "stablehlo.select"(%75, %122, %125) : (tensor<9xi1>, tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    "stablehlo.custom_call"(%126, %51) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<9xf64>, tensor<9xf64>) -> ()
    "func.return"(%126) : (tensor<9xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<9xf64>, tensor<9xf64>, tensor<9xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %47 = "stablehlo.constant"() <{value = dense<[-1.600000e+00, -1.400000e+00, -1.000000e+00, 0.000000e+00, 1.000000e-01, 3.000000e-01, 1.000000e+00, 1.400000e+00, 1.600000e+00]> : tensor<9xf64>}> : () -> tensor<9xf64>
    %48 = "stablehlo.constant"() <{value = dense<[-1.600000e+00, 1.400000e+00, 1.000000e+00, 0.000000e+00, 2.000000e-01, 1.000000e-01, 1.000000e+00, 1.400000e+00, -1.600000e+00]> : tensor<9xf64>}> : () -> tensor<9xf64>
    %49 = "stablehlo.constant"() <{value = dense<[1.000000e+00, -1.000000e+00, 2.000000e+00, 1.000000e+00, 3.000000e-01, 3.000000e-01, -1.000000e+00, 2.400000e+00, 1.600000e+00]> : tensor<9xf64>}> : () -> tensor<9xf64>
    "func.return"(%47, %48, %49) : (tensor<9xf64>, tensor<9xf64>, tensor<9xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<9xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %46 = "stablehlo.constant"() <{value = dense<[0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000, 0.62284335472031604, 0.19461038618229984, 0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000]> : tensor<9xf64>}> : () -> tensor<9xf64>
    "func.return"(%46) : (tensor<9xf64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "remainder", sym_visibility = "private"}> ({
  ^bb0(%arg9: tensor<9xi64>, %arg10: tensor<9xi64>):
    %26 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %27 = "stablehlo.broadcast_in_dim"(%26) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %28 = "stablehlo.compare"(%arg10, %27) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %29 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %30 = "stablehlo.broadcast_in_dim"(%29) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %31 = "func.call"(%28, %30, %arg10) <{callee = @_where}> : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %32 = "stablehlo.remainder"(%arg9, %31) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %33 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %34 = "stablehlo.broadcast_in_dim"(%33) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %35 = "stablehlo.compare"(%32, %34) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %36 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %37 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %38 = "stablehlo.compare"(%32, %37) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %39 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %40 = "stablehlo.broadcast_in_dim"(%39) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %41 = "stablehlo.compare"(%31, %40) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %42 = "stablehlo.compare"(%38, %41) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %43 = "stablehlo.and"(%42, %35) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %44 = "stablehlo.add"(%32, %31) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %45 = "stablehlo.select"(%43, %44, %32) : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    "func.return"(%45) : (tensor<9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg6: tensor<9xi1>, %arg7: tensor<9xi64>, %arg8: tensor<9xi64>):
    %25 = "stablehlo.select"(%arg6, %arg7, %arg8) : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    "func.return"(%25) : (tensor<9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "floor_divide", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<9xi64>, %arg5: tensor<9xi64>):
    %4 = "stablehlo.divide"(%arg4, %arg5) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %5 = "stablehlo.sign"(%arg4) : (tensor<9xi64>) -> tensor<9xi64>
    %6 = "stablehlo.sign"(%arg5) : (tensor<9xi64>) -> tensor<9xi64>
    %7 = "stablehlo.compare"(%5, %6) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %8 = "stablehlo.remainder"(%arg4, %arg5) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %9 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %11 = "stablehlo.compare"(%8, %10) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi1>
    %12 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %13 = "stablehlo.broadcast_in_dim"(%12) <{broadcast_dimensions = array<i64>}> : (tensor<i1>) -> tensor<9xi1>
    %14 = "stablehlo.compare"(%7, %13) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %15 = "stablehlo.convert"(%14) : (tensor<9xi1>) -> tensor<9xi1>
    %16 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %17 = "stablehlo.broadcast_in_dim"(%16) <{broadcast_dimensions = array<i64>}> : (tensor<i1>) -> tensor<9xi1>
    %18 = "stablehlo.compare"(%11, %17) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %19 = "stablehlo.convert"(%18) : (tensor<9xi1>) -> tensor<9xi1>
    %20 = "stablehlo.and"(%15, %19) : (tensor<9xi1>, tensor<9xi1>) -> tensor<9xi1>
    %21 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %22 = "stablehlo.broadcast_in_dim"(%21) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<9xi64>
    %23 = "stablehlo.subtract"(%4, %22) : (tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    %24 = "func.call"(%20, %23, %4) <{callee = @_where_0}> : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    "func.return"(%24) : (tensor<9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_where_0", sym_visibility = "private"}> ({
  ^bb0(%arg1: tensor<9xi1>, %arg2: tensor<9xi64>, %arg3: tensor<9xi64>):
    %3 = "stablehlo.select"(%arg1, %arg2, %arg3) : (tensor<9xi1>, tensor<9xi64>, tensor<9xi64>) -> tensor<9xi64>
    "func.return"(%3) : (tensor<9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<9xf64>) -> tensor<9xf64>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<9xf64>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<9xf64>
    %2 = "stablehlo.divide"(%1, %arg0) : (tensor<9xf64>, tensor<9xf64>) -> tensor<9xf64>
    "func.return"(%2) : (tensor<9xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

